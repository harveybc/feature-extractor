"""Unattended FS-GEN batch pipeline: generator fit -> calibration -> knockoffs only if calibrated.

Per (batch, fold) cell, for every feature of the ps2_batch.v1 batch:
  1. fit the conditional AR generator on the fold fit range (TRAIN only);
  2. synthesize the inner validation range (inside TRAIN) and run the calibration gates;
  3. for CALIBRATED features, build knockoffs x~_t = mu_t + sigma_t * copula(z~_t) where z~ are
     second-order Gaussian group knockoffs of the standardized innovations (groups = lane B
     dependence clusters), check exchangeability gates, then run the knockoff+ filter against the
     14 target cells (Y_s h1..h6, Y_l h24..h144, Y_b s6/l144) at the declared q;
  4. everything else is NOT_CALIBRATED (fail-closed).
Resumable: each cell is persisted as JSON; progress.json is rewritten after every feature.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from app import univariate_temporal as U
from app.fs_gen import calibration as C
from app.fs_gen import contracts as K
from app.fs_gen import generator as G
from app.fs_gen import knockoffs as KO
from app.univariate_temporal_pilot import read_ps2_batch

RUN_SCHEMA = "fs_gen_run.v1"
Y_B_MAP = {2: 1.0, 0: -1.0, 1: 0.0}  # TP first -> +1, SL first -> -1, timeout -> 0, -1 (no support) excluded

FOLD_FIELDS = ["batch", "feature_id", "fold_id", "generator", "order", "n_fit_rows", "n_val_observed_rows",
               "acf_max_abs_diff_lags_1_48", "psd_log_ratio_rmse", "ks_marginal", "tail_ratio_q99_abs",
               "kurtosis_ratio", "regime_coverage", "prefix_invariance", "generator_state", "failing_gates",
               "second_moment_gap", "swap_classifier_auc_batch", "conditional_independence_abs_z_max",
               "knockoff_state", "knockoff_reason", "group_members", "knockoff_selected_cells", "knockoff_W_json",
               "knockoff_T_json", "fdr_q", "seed", "cost_s"]
FEATURE_FIELDS = ["feature_id", "batch", "n_folds", "n_folds_calibrated", "calibration_state", "failing_gates_union",
                  "knockoff_state", "n_folds_knockoff_run", "knockoff_selected_cells_majority",
                  "knockoff_selected_cells_any", "selection_jaccard_mean", "fdr_q", "generator", "seed",
                  "series_sha256", "cost_s", "reason"]


def _atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


def _peak_rss_mb() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    b = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round((a + b) / 1024.0, 1)


def _code_sha(root: str) -> str:
    h = hashlib.sha256()
    for name in sorted(os.listdir(root)):
        if name.endswith(".py"):
            h.update(name.encode())
            h.update(open(os.path.join(root, name), "rb").read())
    return h.hexdigest()


def _target_cells(targets: dict, man: dict) -> Dict[str, np.ndarray]:
    """14 cells as float vectors with NaN = no support."""
    cells: Dict[str, np.ndarray] = {}
    cols = man.get("target_columns") or {}
    ys, yl, yb = targets.get("Y_s"), targets.get("Y_l"), targets.get("Y_b")
    if ys is not None:
        names = cols.get("Y_s") or [f"Y_s_h{h}" for h in range(1, ys.shape[1] + 1)]
        for i, nm in enumerate(names[: ys.shape[1]]):
            cells[nm.split(" ")[0]] = np.asarray(ys[:, i], np.float64)
    if yl is not None:
        names = cols.get("Y_l") or [f"Y_l_h{24 * (i + 1)}" for i in range(yl.shape[1])]
        for i, nm in enumerate(names[: yl.shape[1]]):
            cells[nm.split(" ")[0]] = np.asarray(yl[:, i], np.float64)
    if yb is not None:
        yb = np.asarray(yb)
        if yb.ndim == 1:
            yb = yb[:, None]
        names = cols.get("Y_b") or [f"Y_b_{i}" for i in range(yb.shape[1])]
        for i, nm in enumerate(names[: yb.shape[1]]):
            v = np.full(yb.shape[0], np.nan)
            for k, val in Y_B_MAP.items():
                v[yb[:, i] == k] = val
            if yb.shape[1] == 1 and cols.get("Y_b") is None:  # synthetic batches encode 0/1
                v = np.where(yb[:, 0] >= 0, yb[:, 0].astype(float), np.nan)
            cells[nm.split(" ")[0]] = v
    return cells


def _range_idx(ts: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.nonzero((ts >= lo) & (ts <= hi))[0]


def _subsample(idx: np.ndarray, max_rows: Optional[int]) -> np.ndarray:
    if max_rows is None or idx.size <= max_rows:
        return idx
    return idx[np.linspace(0, idx.size - 1, max_rows).astype(int)]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_cell(batch: dict, fold: dict, groups_clusters: Sequence[Sequence[str]], seed: int, order: int, rule: dict,
             max_fit_rows: Optional[int], progress_cb=None) -> List[dict]:
    man = batch["manifest"]
    series, targets = batch["series"], batch["targets"]
    ts = np.asarray(series["timestamps"], np.int64)
    K.assert_train_only(ts, int(man["train_end_ts"]))
    features: List[str] = list(man["features"])
    fit_idx_all = _range_idx(ts, fold["fit"][0], fold["fit"][1])
    val_idx = _range_idx(ts, fold["val"][0], fold["val"][1])
    if val_idx.size == 0 or fit_idx_all.size == 0:
        raise U.FoldScopeError(f"fold {fold['fold_id']} has an empty range on this grid")
    if int(ts[val_idx].max()) > int(man["train_end_ts"]):
        raise U.FoldScopeError("validation range leaves TRAIN")
    fit_idx_all = fit_idx_all[fit_idx_all >= order]
    v0, v1 = int(val_idx[0]), int(val_idx[-1]) + 1
    q = float(rule["fdr_q"])
    rows: Dict[str, dict] = {}
    gens: Dict[str, G.ConditionalARGenerator] = {}
    t_cell = time.time()
    for f in features:
        t0 = time.time()
        x = np.asarray(series[f"x__{f}"], np.float64)
        obs = np.isfinite(x)
        row = {k: "" for k in FOLD_FIELDS}
        row.update(batch=man["batch_id"], feature_id=f, fold_id=fold["fold_id"], generator="conditional_heteroskedastic_ar",
                   order=order, fdr_q=q, seed=seed, knockoff_state="NOT_CALIBRATED", knockoff_selected_cells="")
        try:
            g = G.ConditionalARGenerator(order=order, seed=seed).fit(ts, x, obs, fit_idx_all)
            synth = g.synthesize(ts, x, obs, v0, v1)
            pi = G.prefix_invariance_check(g, ts, x, obs, v0, v1)
            m = C.generator_diagnostics(x[v0:v1], synth, obs[v0:v1], prefix_invariance=pi)
            state, failing = C.decide_generator(m, rule)
            row.update(n_fit_rows=g.n_fit, **{k: m[k] for k in m})
            row.update(generator_state=state, failing_gates=";".join(failing))
            if state == "CALIBRATED":
                gens[f] = g
            else:
                row["knockoff_reason"] = "GENERATOR_NOT_CALIBRATED"
        except U.ContractError as e:
            row.update(generator_state="NOT_CALIBRATED", failing_gates=f"fit_error:{type(e).__name__}",
                       knockoff_reason="GENERATOR_FIT_REFUSED:" + str(e)[:120])
        row["cost_s"] = round(time.time() - t0, 3)
        rows[f] = row
        if progress_cb:
            progress_cb(f, "generator")

    # ---- knockoffs on the fit rows for the calibrated features (fail-closed otherwise)
    cal_feats = [f for f in features if f in gens]
    if cal_feats and targets is not None:
        rng = np.random.default_rng(seed)
        fit_idx = _subsample(fit_idx_all, max_fit_rows)
        n, p = fit_idx.size, len(cal_feats)
        X = np.zeros((n, p))
        Xk = np.zeros((n, p))
        for j, f in enumerate(cal_feats):
            g = gens[f]
            x = np.asarray(series[f"x__{f}"], np.float64)
            obs = np.isfinite(x)
            mu, sig, ok = g.conditional_moments(ts, x, obs, fit_idx)
            xn = np.where(ok, (np.where(obs[fit_idx], x[fit_idx], 0.0) - g.mean) / g.std, 0.0)
            X[:, j] = xn
            Xk[:, j] = np.where(ok, mu, 0.0)  # filled below with sigma * copula(z~)
        # standardized innovations matrix (0 where not ok, mask-symmetric for X and X~)
        Z = np.zeros((n, p))
        OK = np.zeros((n, p), bool)
        SIG = np.ones((n, p))
        for j, f in enumerate(cal_feats):
            g = gens[f]
            x = np.asarray(series[f"x__{f}"], np.float64)
            z, ok = g.innovations(ts, x, np.isfinite(x), fit_idx)
            _, sig, _ = g.conditional_moments(ts, x, np.isfinite(x), fit_idx)
            Z[:, j], OK[:, j], SIG[:, j] = z, ok, sig
        Zc = Z - Z.mean(0)
        Sigma = np.cov(Zc, rowvar=False) if p > 1 else np.array([[float(Zc.var())]])
        Sigma = Sigma + 1e-4 * np.eye(p)
        groups = KO.groups_from_clusters(cal_feats, groups_clusters)
        S = KO.group_equicorrelated_S(Sigma, groups)
        Zk = KO.gaussian_knockoffs(Zc, Sigma, S, rng) + Z.mean(0)
        for j, f in enumerate(cal_feats):
            emp = Z[OK[:, j], j]
            zk = KO.copula_map(Zk[:, j], emp) if emp.size >= 50 else Zk[:, j]
            Xk[:, j] = np.where(OK[:, j], Xk[:, j] + SIG[:, j] * zk, 0.0)
        Xc, Xkc = X - X.mean(0), Xk - X.mean(0)
        gap = C.second_moment_gap(Xc, Xkc) if p > 1 else np.zeros(p)
        auc = C.swap_classifier_auc(Xc, Xkc, rng) if p > 1 else 0.5
        kr = rule["knockoff"]
        auc_ok = kr["swap_classifier_auc_batch"]["min"] <= auc <= kr["swap_classifier_auc_batch"]["max"]
        cells = _target_cells(targets, man)
        ci_max = np.zeros(p)
        W_by_cell: Dict[str, np.ndarray] = {}
        T_by_cell: Dict[str, float] = {}
        for cname, yfull in cells.items():
            y = yfull[fit_idx]
            keep = np.isfinite(y)
            if keep.sum() < 300 or np.std(y[keep]) == 0:
                continue
            ci = C.conditional_independence_abs_z(y[keep], Xc[keep], Xkc[keep])
            ci_max = np.maximum(ci_max, ci)
            W = KO.group_lasso_diff_stat(Xc[keep], Xkc[keep], y[keep], groups, seed=seed)
            W_by_cell[cname] = W
            T_by_cell[cname] = KO.knockoff_threshold(W, q, offset=1)
            if progress_cb:
                progress_cb(cname, "knockoff")
        gidx = {j: gi for gi, g in enumerate(groups) for j in g}
        for j, f in enumerate(cal_feats):
            row = rows[f]
            row.update(second_moment_gap=round(float(gap[j]), 5), swap_classifier_auc_batch=round(auc, 4),
                       conditional_independence_abs_z_max=round(float(ci_max[j]), 3),
                       group_members=";".join(cal_feats[k] for k in groups[gidx[j]]))
            reasons = []
            if gap[j] > kr["second_moment_gap_per_feature"]["max"]:
                reasons.append("second_moment_gap")
            if not auc_ok:
                reasons.append("swap_classifier_auc_batch")
            if ci_max[j] > kr["conditional_independence_abs_z"]["max"]:
                reasons.append("conditional_independence_abs_z")
            if not W_by_cell:
                reasons.append("no_target_cell_with_support")
            if reasons:
                row.update(knockoff_state="NOT_CALIBRATED", knockoff_reason=";".join(reasons))
                continue
            sel = [c for c, W in W_by_cell.items() if W[gidx[j]] >= T_by_cell[c]]
            row.update(knockoff_state="RUN", knockoff_reason="", knockoff_selected_cells=";".join(sel),
                       knockoff_W_json=json.dumps({c: round(float(W[gidx[j]]), 6) for c, W in W_by_cell.items()}),
                       knockoff_T_json=json.dumps({c: (None if not np.isfinite(T) else round(T, 6)) for c, T in T_by_cell.items()}))
    elif cal_feats and targets is None:
        for f in cal_feats:
            rows[f].update(knockoff_reason="NO_TARGETS_IN_BATCH")
    cell_cost = time.time() - t_cell
    out = [rows[f] for f in features]
    for r in out:
        r["cell_cost_s"] = round(cell_cost, 1)
    return out


def summarize_features(fold_rows: List[dict], rule: dict, series_sha: Dict[str, str]) -> List[dict]:
    maj = int(rule["fold_majority"])
    by: Dict[str, List[dict]] = {}
    for r in fold_rows:
        by.setdefault(r["feature_id"], []).append(r)
    out = []
    for f, rs in sorted(by.items()):
        n = len(rs)
        ncal = sum(r["generator_state"] == "CALIBRATED" for r in rs)
        nrun = sum(r["knockoff_state"] == "RUN" for r in rs)
        fails = sorted({g for r in rs for g in str(r.get("failing_gates", "")).split(";") if g})
        sels = [set(str(r["knockoff_selected_cells"]).split(";")) - {""} for r in rs if r["knockoff_state"] == "RUN"]
        counts: Dict[str, int] = {}
        for s in sels:
            for c in s:
                counts[c] = counts.get(c, 0) + 1
        cal_state = "CALIBRATED" if ncal >= maj else "NOT_CALIBRATED"
        ko_state = "RUN" if (cal_state == "CALIBRATED" and nrun >= maj) else "NOT_CALIBRATED"
        jac = [_jaccard(sels[a], sels[b]) for a in range(len(sels)) for b in range(a + 1, len(sels))]
        out.append({
            "feature_id": f, "batch": rs[0]["batch"], "n_folds": n, "n_folds_calibrated": ncal,
            "calibration_state": cal_state, "failing_gates_union": ";".join(fails), "knockoff_state": ko_state,
            "n_folds_knockoff_run": nrun,
            "knockoff_selected_cells_majority": ";".join(sorted(c for c, k in counts.items() if k >= maj)) if ko_state == "RUN" else "",
            "knockoff_selected_cells_any": ";".join(sorted(counts)) if ko_state == "RUN" else "",
            "selection_jaccard_mean": round(float(np.mean(jac)), 4) if jac else "",
            "fdr_q": rule["fdr_q"], "generator": rs[0]["generator"], "seed": rs[0]["seed"],
            "series_sha256": series_sha.get(rs[0]["batch"], ""), "cost_s": round(sum(float(r["cost_s"] or 0) for r in rs), 2),
            "reason": "" if ko_state == "RUN" else ("GENERATOR_NOT_CALIBRATED_ON_MAJORITY" if cal_state != "CALIBRATED" else "KNOCKOFF_EXCHANGEABILITY_NOT_CALIBRATED_ON_MAJORITY"),
        })
    return out


def denominator_rows(denominator: Sequence[str], feature_rows: List[dict]) -> List[dict]:
    """One row per candidate of the 366 denominator; fail-closed for features without a series."""
    by = {r["feature_id"]: r for r in feature_rows}
    out = []
    for f in denominator:
        if f in by:
            out.append(dict(by[f]))
            continue
        row = {k: "" for k in FEATURE_FIELDS}
        row["feature_id"] = f
        if f.startswith("cal."):
            row.update(calibration_state="NOT_APPLICABLE", knockoff_state="NOT_APPLICABLE",
                       reason="CALENDAR_CONDITIONING_NOT_EXTRACTOR_SERIES")
        else:
            row.update(calibration_state="NOT_EVALUATED", knockoff_state="NOT_CALIBRATED", reason="NO_SERIES_IN_PS2_BATCH")
        out.append(row)
    return out


def _write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    os.replace(tmp, path)


def run_batches(batch_dirs: Sequence[str], out_dir: str, groups_by_batch: Dict[str, Dict[str, list]], seed: int,
                order: int, rule: dict, max_fit_rows: Optional[int] = 20000, denominator: Optional[Sequence[str]] = None) -> dict:
    os.makedirs(os.path.join(out_dir, "cells"), exist_ok=True)
    prog_path = os.path.join(out_dir, "progress.json")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t_start = time.time()
    batches = []
    for d in batch_dirs:
        b = read_ps2_batch(d)
        K.assert_train_only(b["series"]["timestamps"], int(b["manifest"]["train_end_ts"]))
        batches.append((d, b))
    plan = [(d, b, fold) for d, b in batches for fold in b["manifest"]["folds"]]
    prog = {"schema": "fs_gen_progress.v1", "lane": "FS-GEN", "state": "RUNNING", "started_utc": started,
            "updated_utc": started, "cells_total": len(plan), "cells_done": 0, "features_total": sum(len(b["manifest"]["features"]) for _, b in batches),
            "current": None, "cost_s": 0.0, "peak_rss_mb": _peak_rss_mb(), "seed": seed, "order": order,
            "fdr_q": rule["fdr_q"], "batches": {b["manifest"]["batch_id"]: {"n_features": len(b["manifest"]["features"]), "folds_done": []} for _, b in batches},
            "generator_states": {}, "knockoff_states": {}}

    def flush(current=None):
        prog["current"] = current
        prog["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        prog["cost_s"] = round(time.time() - t_start, 1)
        prog["peak_rss_mb"] = _peak_rss_mb()
        _atomic_json(prog_path, prog)

    flush()
    fold_rows: List[dict] = []
    new_cells = 0
    for d, b, fold in plan:
        bid = b["manifest"]["batch_id"]
        cell_path = os.path.join(out_dir, "cells", f"{bid}__{fold['fold_id']}.json")
        if os.path.isfile(cell_path):
            rows = json.load(open(cell_path))["rows"]
        else:
            clusters = (groups_by_batch.get(bid) or {}).get(fold["fold_id"]) or []
            cb = lambda item, stage, _b=bid, _f=fold["fold_id"]: flush({"batch": _b, "fold": _f, "stage": stage, "item": item})
            rows = run_cell(b, fold, clusters, seed, order, rule, max_fit_rows, progress_cb=cb)
            _atomic_json(cell_path, {"schema": "fs_gen_cell.v1", "batch": bid, "fold": fold, "rows": rows,
                                     "series_sha256": b["manifest"]["series"]["sha256"],
                                     "targets_sha256": (b["manifest"].get("targets") or {}).get("sha256"),
                                     "n_clusters": len(clusters), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            new_cells += 1
        fold_rows.extend(rows)
        prog["cells_done"] += 1
        prog["batches"][bid]["folds_done"].append(fold["fold_id"])
        gs, ks = {}, {}
        for r in fold_rows:
            gs[r["generator_state"]] = gs.get(r["generator_state"], 0) + 1
            ks[r["knockoff_state"]] = ks.get(r["knockoff_state"], 0) + 1
        prog["generator_states"], prog["knockoff_states"] = gs, ks
        flush()
    series_sha = {b["manifest"]["batch_id"]: b["manifest"]["series"]["sha256"] for _, b in batches}
    feature_rows = summarize_features(fold_rows, rule, series_sha)
    _write_csv(os.path.join(out_dir, "generative_evidence_folds.csv"), fold_rows, FOLD_FIELDS + ["cell_cost_s"])
    _write_csv(os.path.join(out_dir, "generative_evidence_features.csv"), feature_rows, FEATURE_FIELDS)
    denom_rows = denominator_rows(denominator, feature_rows) if denominator else feature_rows
    _write_csv(os.path.join(out_dir, "generative_evidence.csv"), denom_rows, FEATURE_FIELDS)
    manifest = {
        "schema": RUN_SCHEMA, "lane": "FS-GEN", "contract": K.CONTRACT, "rule": rule, "seed": seed, "order": order,
        "max_fit_rows": max_fit_rows, "batches": [{"dir": d, "batch_id": b["manifest"]["batch_id"], "manifest_sha256": b["manifest_sha256"],
                                                   "series_sha256": b["manifest"]["series"]["sha256"],
                                                   "targets_sha256": (b["manifest"].get("targets") or {}).get("sha256"),
                                                   "n_features": len(b["manifest"]["features"]), "train_end_ts": b["manifest"]["train_end_ts"]} for d, b in batches],
        "code_sha256": _code_sha(os.path.dirname(os.path.abspath(__file__))),
        "python": platform.python_version(), "numpy": np.__version__, "cost_s": round(time.time() - t_start, 1),
        "peak_rss_mb": _peak_rss_mb(), "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "outputs": {n: U.sha256_file(os.path.join(out_dir, n)) for n in
                    ("generative_evidence.csv", "generative_evidence_features.csv", "generative_evidence_folds.csv")},
        "counts": {"fold_rows": len(fold_rows), "feature_rows": len(feature_rows), "denominator_rows": len(denom_rows),
                   "calibrated_features": sum(r["calibration_state"] == "CALIBRATED" for r in feature_rows),
                   "knockoff_run_features": sum(r["knockoff_state"] == "RUN" for r in feature_rows)},
    }
    _atomic_json(os.path.join(out_dir, "run_manifest.json"), manifest)
    prog["state"] = "DONE"
    prog["counts"] = manifest["counts"]
    flush()
    return {"fold_rows": fold_rows, "feature_rows": feature_rows, "denominator_rows": denom_rows, "new_cells": new_cells,
            "manifest": manifest}


def load_lane_b_groups(lane_b_dirs: Sequence[str]) -> Dict[str, Dict[str, list]]:
    out: Dict[str, Dict[str, list]] = {}
    for d in lane_b_dirs:
        p = os.path.join(d, "ps2_manifest.json")
        if not os.path.isfile(p):
            continue
        m = json.load(open(p))
        out[m["batch_id"]] = {fid: fc.get("dependence", []) for fid, fc in (m.get("fold_clusters") or {}).items()}
    return out


def load_denominator(path: str) -> List[str]:
    with open(path, newline="") as f:
        return [r["feature_id"] for r in csv.DictReader(f)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch_dirs", required=True, help="comma-separated ps2_batch.v1 directories")
    ap.add_argument("--lane_b_dirs", default="", help="comma-separated lane B batch dirs holding ps2_manifest.json (fold_clusters)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--denominator_csv", default="", help="coverage_by_batch_feature.csv (366 rows) for the fail-closed denominator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", type=int, default=24)
    ap.add_argument("--max_fit_rows", type=int, default=20000)
    ap.add_argument("--fdr_q", type=float, default=0.10)
    a = ap.parse_args(argv)
    rule = C.default_rule()
    rule["fdr_q"] = a.fdr_q
    groups = load_lane_b_groups([d for d in a.lane_b_dirs.split(",") if d])
    denom = load_denominator(a.denominator_csv) if a.denominator_csv else None
    res = run_batches([d for d in a.batch_dirs.split(",") if d], a.out_dir, groups, a.seed, a.order, rule,
                      max_fit_rows=a.max_fit_rows, denominator=denom)
    print(json.dumps(res["manifest"]["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
