"""Pilot runner: univariate temporal extractor families over one PS2 batch (lanes E/F).

Consumed batch contract `ps2_batch.v1` (declared by lane D 2026-10-03; lane B produces it):

    batch_NNN/
      batch_manifest.json
        {"schema": "ps2_batch.v1", "batch_id": "batch_NNN", "asset": "EURUSD",
         "sampling_period_seconds": 3600,
         "series":  {"file": "series.npz",  "sha256": "<64 hex>"},
         "targets": {"file": "targets.npz", "sha256": "<64 hex>"} | null,
         "features": ["<feature_id>", ...],          # PS2 survivors + exploratory sample
         "known_calendar": ["<name>", ...],          # optional; published-at-t columns
         "train_end_ts": <int epoch s>,               # last TRAIN timestamp; nothing later is read
         "folds": [{"fold_id": "f0", "split": "train",
                    "fit": [first_anchor_ts, last_anchor_ts],
                    "val": [first_anchor_ts, last_anchor_ts]}, ...]}
      series.npz   timestamps int64 (N,) UTC seconds on a regular grid of sampling_period_seconds;
                   x__<feature_id> float32 (N,), NaN = not observed;
                   cal__<name> float32 (N,) and calpub__<name> int64 (N,) for known_calendar columns
      targets.npz  timestamps int64 (N,) == series timestamps; Y_s float (N,6) h=1..6;
                   Y_l float (N,6) h=24..144; Y_b int (N,) or (N,k), -1 = no support; NaN = no support.
                   Targets supervise probes only; they are never encoder inputs.

Every digest is verified before any fit; a mismatch refuses the whole batch.
Rows after train_end_ts are never windowed (only fold anchors are, all inside TRAIN).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Optional

import numpy as np

from app import temporal_extractor_metrics as M
from app import univariate_temporal as U

BATCH_SCHEMA = "ps2_batch.v1"
RUN_SCHEMA = "ut_pilot_run.v1"


class BatchContractError(U.ContractError):
    pass


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_ps2_batch(batch_dir: str) -> dict:
    mpath = os.path.join(batch_dir, "batch_manifest.json")
    if not os.path.isfile(mpath):
        raise BatchContractError(f"missing {mpath}")
    raw = open(mpath, "rb").read()
    man = json.loads(raw)
    if man.get("schema") != BATCH_SCHEMA:
        raise BatchContractError(f"batch schema {man.get('schema')!r} != {BATCH_SCHEMA}")
    U.check_no_target(man.get("features", []))
    out = {"manifest": man, "manifest_sha256": _sha_bytes(raw)}
    for part in ("series", "targets"):
        spec = man.get(part)
        if spec is None:
            if part == "series":
                raise BatchContractError("series is required")
            out[part] = None
            continue
        p = os.path.join(batch_dir, spec["file"])
        if not os.path.isfile(p) or U.sha256_file(p) != spec.get("sha256"):
            raise BatchContractError(f"{part} file missing or digest mismatch")
        with np.load(p, allow_pickle=False) as z:
            out[part] = {k: z[k] for k in z.files}
    s = out["series"]
    ts = np.asarray(s["timestamps"], np.int64)
    period = int(man["sampling_period_seconds"])
    if ts.ndim != 1 or ts.size < 2 or np.any(np.diff(ts) != period):
        raise BatchContractError("series timestamps must be a regular grid at sampling_period_seconds "
                                 "(different frequencies need a separate causal alignment transform)")
    for f in man["features"]:
        if f"x__{f}" not in s:
            raise BatchContractError(f"feature {f!r} missing from series")
    if out["targets"] is not None:
        if not np.array_equal(out["targets"]["timestamps"], ts):
            raise BatchContractError("targets grid differs from series grid")
    if not man.get("folds"):
        raise BatchContractError("at least one TRAIN fold is required")
    return out


def write_synthetic_ps2_batch(batch_dir: str, n: int = 2000, features=("feat_a", "feat_b"), seed: int = 0,
                              window: int = 24, start_ts: int = 1704067200) -> dict:
    """Tiny synthetic batch honoring ps2_batch.v1 (tests and dry runs only)."""
    rng = np.random.default_rng(seed)
    os.makedirs(batch_dir, exist_ok=True)
    ts = start_ts + 3600 * np.arange(n, dtype=np.int64)
    hour = (ts // 3600) % 24
    series = {"timestamps": ts}
    price = None
    for i, f in enumerate(features):
        e = rng.normal(size=n)
        v = np.zeros(n)
        for t in range(1, n):
            v[t] = 0.9 * v[t - 1] + e[t]
        v = v + 0.5 * np.sin(2 * np.pi * hour / 24) + 100 * (i == 0)
        if price is None:
            price = v.copy()
        v = v.astype(np.float32)
        v[rng.random(n) < 0.05] = np.nan
        series[f"x__{f}"] = v
    def fwd(h):
        r = np.full(n, np.nan)
        r[:-h] = price[h:] - price[:-h]
        return r
    ys = np.stack([fwd(h) for h in range(1, 7)], 1)
    yl = np.stack([fwd(h) for h in range(24, 145, 24)], 1)
    yb = np.where(np.isfinite(ys[:, 5]), (ys[:, 5] > 0).astype(np.int64), -1)
    targets = {"timestamps": ts, "Y_s": ys, "Y_l": yl, "Y_b": yb}
    np.savez(os.path.join(batch_dir, "series.npz"), **series)
    np.savez(os.path.join(batch_dir, "targets.npz"), **targets)
    g = window + 6
    a = window + 16
    q = (n - 1 - a - 2 * g) // 4
    folds = [{"fold_id": "f0", "split": "train", "fit": [int(ts[a]), int(ts[a + q])],
              "val": [int(ts[a + q + g]), int(ts[a + 2 * q])]},
             {"fold_id": "f1", "split": "train", "fit": [int(ts[a]), int(ts[a + 2 * q])],
              "val": [int(ts[a + 2 * q + g]), int(ts[a + 3 * q + g])]}]
    man = {"schema": BATCH_SCHEMA, "batch_id": os.path.basename(os.path.normpath(batch_dir)),
           "asset": "SYNTHETIC", "sampling_period_seconds": 3600,
           "series": {"file": "series.npz", "sha256": U.sha256_file(os.path.join(batch_dir, "series.npz"))},
           "targets": {"file": "targets.npz", "sha256": U.sha256_file(os.path.join(batch_dir, "targets.npz"))},
           "features": list(features), "known_calendar": [], "train_end_ts": int(ts[-1]), "folds": folds}
    with open(os.path.join(batch_dir, "batch_manifest.json"), "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    return man


def _cgroup_peak_bytes() -> Optional[int]:
    try:
        rel = open("/proc/self/cgroup").read().strip().split("::")[-1]
        p = os.path.join("/sys/fs/cgroup", rel.lstrip("/"), "memory.peak")
        return int(open(p).read().strip())
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _even_subset(idx: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0 or idx.size <= cap:
        return idx
    return idx[np.unique(np.linspace(0, idx.size - 1, cap).round().astype(int))]


def run_pilot(a) -> dict:
    t_start = time.perf_counter()
    batch = read_ps2_batch(a.batch_dir)  # every digest verified before any fit
    man, s, tg = batch["manifest"], batch["series"], batch["targets"]
    ts = np.asarray(s["timestamps"], np.int64)
    train_end = int(man["train_end_ts"])
    folds = [U.FoldSpec(f["fold_id"], f["split"], tuple(f["fit"]), tuple(f["val"])) for f in man["folds"]]
    for f in folds:
        U.validate_fold(f, ts, train_end, a.window)
    features = a.features.split(",") if a.features else list(man["features"])
    unknown = set(features) - set(man["features"])
    if unknown:
        raise BatchContractError(f"features not in batch: {sorted(unknown)}")
    families = a.families.split(",")
    for fam in families:
        if fam not in U.FAMILIES or U.FAMILIES[fam].status != "IMPLEMENTED":
            raise U.ContractError(f"family {fam!r} not runnable ({U.FAMILIES.get(fam)})")
    cal = U.calendar_features(ts)
    kc = man.get("known_calendar") or []
    if kc:
        cal = np.concatenate([cal, U.known_calendar_columns(
            ts, {k: (s[f"cal__{k}"], s[f"calpub__{k}"]) for k in kc})], axis=1)
    cfg = U.ArchConfig(window=a.window, calendar_dim=cal.shape[1], latent_dim=a.latent_dim, filters=a.filters,
                       kernel_size=a.kernel_size, dilations=tuple(int(d) for d in a.dilations.split(",")),
                       decoder_filters=a.filters)
    es = U.EarlyStopConfig(max_epochs=a.max_epochs, patience=a.patience, min_delta=a.min_delta)
    lags = tuple(int(l) for l in a.probe_lags.split(","))
    keep = max(lags) + 1
    os.makedirs(a.out_dir, exist_ok=True)
    res_path = os.path.join(a.out_dir, "results.jsonl")
    open(res_path, "w").close()

    def emit(row):
        with open(res_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True, default=float) + "\n")

    ref_fold = folds[-1]
    targets = {}
    if tg is not None:
        targets = {k: v for k, v in tg.items() if k != "timestamps"}
    for feat in features:
        x = np.asarray(s[f"x__{feat}"], np.float32)
        obs = np.isfinite(x)
        ref_lat = {fam: {} for fam in families}
        probe_losses = {}
        for fold in folds:
            fi = _even_subset(U.fold_anchor_indices(ts, fold.fit), a.max_fit_windows)
            vi = _even_subset(U.fold_anchor_indices(ts, fold.val), a.max_val_windows)
            norm = U.Normalization.fit(x, obs, U.covered_indices(fi, a.window))
            fit_b = U.make_windows(ts, x, obs, fi, a.window, norm, calendar=cal)
            val_b = U.make_windows(ts, x, obs, vi, a.window, norm, calendar=cal)
            ri = _even_subset(U.fold_anchor_indices(ts, ref_fold.val), a.max_ref_windows)
            ref_b = U.make_windows(ts, x, obs, ri, a.window, norm, calendar=cal)
            fit_ids_sha = U.row_ids_sha256(fit_b.row_ids)
            thr = float(np.quantile(np.abs(fit_b.signal[fit_b.observed_mask > 0]), 0.99))
            reps, cal_tail = {}, np.concatenate([fit_b.calendar, val_b.calendar])[:, -keep:, :]
            for fam in families:
                ext = U.make_extractor(fam, cfg, a.seed, learning_rate=a.learning_rate, batch_size=a.batch_size,
                                       **({"corruption": {"type": "gaussian_observed", "sigma": a.dae_sigma}}
                                          if fam == "dae" else {}))
                rep = ext.fit(fit_b, val_b, es)
                t0 = time.perf_counter()
                z_val = ext.encode(val_b)
                lat_ms = 1000 * (time.perf_counter() - t0) / max(len(val_b), 1)
                z_fit = ext.encode(fit_b)
                reps[fam] = np.concatenate([z_fit, z_val])[:, -keep:, :]
                ref_lat[fam][fold.fold_id] = ext.encode(ref_b)
                rec = M.reconstruction_metrics(val_b.signal, ext.reconstruct(val_b), val_b.observed_mask,
                                               norm.mean, norm.std, thr)
                donor = None
                if ext.encoder is not None:
                    ddir = os.path.join(a.out_dir, "donors", feat, fold.fold_id, fam)
                    scope = U.TrainScope("TRAIN_ONLY", "train", fold.fold_id, int(fit_b.anchor_ts[0]),
                                         int(fit_b.anchor_ts[-1]), man["series"]["sha256"], fit_ids_sha)
                    dm = ext.export_donor(ddir, scope, feat, normalization=norm)
                    donor = {"dir": os.path.relpath(ddir, a.out_dir), "weights_sha256": dm["weights_sha256"],
                             "encoder_sha256": dm["encoder_sha256"]}
                rep_small = {k: v for k, v in rep.items() if k != "weights_sha256_by_epoch"}
                emit({"kind": "fold_family", "feature_id": feat, "fold_id": fold.fold_id, "family": fam,
                      "architecture_id": ext.architecture_id, "seed": a.seed, "window": a.window,
                      "latent_dim": ext.latent_dim, "n_fit": len(fit_b), "n_val": len(val_b),
                      "train_row_ids_sha256": fit_ids_sha, "normalization": asdict(norm),
                      "fit_report": rep_small, "reconstruction": rec,
                      "effective_dimension": M.effective_dimension(z_val),
                      "cost": {"fit_wall_seconds": rep.get("fit_wall_seconds", 0.0),
                               "updates": rep.get("updates", 0), "epochs_run": rep.get("epochs_run", 0),
                               "params": int(ext.encoder.count_params()) if ext.encoder is not None else 0,
                               "encode_latency_ms_per_window": lat_ms},
                      "donor": donor})
            if targets:
                nf = len(fit_b)
                rows = M.equal_probes(reps, cal_tail, {k: v[np.concatenate([fi, vi])] for k, v in targets.items()},
                                      np.arange(nf), np.arange(nf, nf + len(vi)), a.ridge_alpha, lags)
                for r in rows:
                    emit(dict(r, kind="probe", feature_id=feat, fold_id=fold.fold_id))
                    if r.get("status") == "MEASURED":
                        v = r["mae"] if r["kind"] == "regression" else r["log_loss"]
                        probe_losses.setdefault((r["representation"], r["target"], r["horizon_index"]), []).append(v)
                for fam in families:
                    if not U.FAMILIES[fam].trainable:
                        continue
                    for (tname, h), d in M.probe_deltas(rows, raw="identity", random="random", trained=fam).items():
                        emit(dict(d, kind="probe_delta", feature_id=feat, fold_id=fold.fold_id,
                                  target=tname, horizon_index=h))
            import keras
            keras.backend.clear_session()  # bound graph growth across features/folds in one process
        emit({"kind": "feature_summary", "feature_id": feat, "reference_fold": ref_fold.fold_id,
              "stability": {fam: M.stability_across_folds(ref_lat[fam]) for fam in families},
              "probe_loss_across_folds": [
                  {"representation": r, "target": t, "horizon_index": h, "n_folds": len(v),
                   "mean": float(np.mean(v)), "std": float(np.std(v))}
                  for (r, t, h), v in sorted(probe_losses.items())]})
    import tensorflow as tf
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    run = {"schema": RUN_SCHEMA, "batch_dir": os.path.abspath(a.batch_dir), "batch_id": man.get("batch_id"),
           "batch_manifest_sha256": batch["manifest_sha256"], "series_sha256": man["series"]["sha256"],
           "targets_sha256": (man.get("targets") or {}).get("sha256"), "features": features, "families": families,
           "folds": [f.fold_id for f in folds], "arch_config": cfg.to_dict(),
           "receptive_field_steps": cfg.receptive_field(), "early_stop": asdict(es), "seed": a.seed,
           "args": vars(a), "code_commit": _git_commit(), "tensorflow": tf.__version__,
           "cpu_count": os.cpu_count(), "machine": platform.machine(),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "peak_rss_bytes": int(ru), "cgroup_peak_bytes": _cgroup_peak_bytes(),
           "wall_seconds": time.perf_counter() - t_start, "results_sha256": U.sha256_file(res_path),
           "status": "COMPLETED"}
    U._atomic_json(os.path.join(a.out_dir, "run_manifest.json"), run)
    return run


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--features", default="", help="comma list; default all batch features")
    p.add_argument("--families", default="identity,random,ae,dae")
    p.add_argument("--window", type=int, default=168)
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--filters", type=int, default=16)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--dilations", default="1,2,4,8,16,32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--dae_sigma", type=float, default=0.1)
    p.add_argument("--max_fit_windows", type=int, default=0, help="0 = all; else evenly spaced subset")
    p.add_argument("--max_val_windows", type=int, default=0)
    p.add_argument("--max_ref_windows", type=int, default=256)
    p.add_argument("--probe_lags", default="0,1,2,23")
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    run = run_pilot(parse(argv))
    print(json.dumps({k: run[k] for k in ("status", "batch_id", "features", "peak_rss_bytes",
                                          "cgroup_peak_bytes", "wall_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
