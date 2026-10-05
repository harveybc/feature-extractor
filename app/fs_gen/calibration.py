"""Calibration diagnostics and the frozen decision rule (fs_closure/fs_gen/CALIBRATION_RULE.json).

Generator gates (real vs synthetic on the inner validation range, outside the generator fit):
temporal fidelity (ACF, spectrum), marginal (KS), tails (q99 ratio, kurtosis ratio), regime
coverage and prefix invariance. Knockoff exchangeability gates: per-feature second-moment gap,
batch swap-classifier AUC and a conditional-independence leak guard against the target.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal, stats

RULE_SCHEMA = "fs_gen.calibration_rule.v1"


def default_rule() -> dict:
    return {
        "schema": RULE_SCHEMA,
        "generator": {
            "acf_max_abs_diff_lags_1_48": {"max": 0.10},
            "psd_log_ratio_rmse": {"max": 0.50},
            "ks_marginal": {"max": 0.10},
            "tail_ratio_q99_abs": {"min": 0.70, "max": 1.40},
            "kurtosis_ratio": {"min": 0.50, "max": 2.00},
            "regime_coverage": {"min": 0.80},
            "prefix_invariance": {"must_be": True},
            "min_val_observed_rows": {"min": 500},
        },
        "knockoff": {
            "second_moment_gap_per_feature": {"max": 0.05},
            "swap_classifier_auc_batch": {"min": 0.45, "max": 0.55},
            "conditional_independence_abs_z": {"max": 4.0},
        },
        "fold_majority": 3,
        "fdr_q": 0.10,
    }


# ------------------------------------------------------------------ helpers
def _acf(v: np.ndarray, nlags: int) -> np.ndarray:
    v = v - v.mean()
    d = float(v @ v)
    if d <= 0:
        return np.zeros(nlags)
    return np.array([float(v[:-k] @ v[k:]) / d for k in range(1, nlags + 1)])


def _rolling_std(v: np.ndarray, w: int) -> np.ndarray:
    if v.size < w:
        return np.array([])
    c1 = np.cumsum(np.insert(v, 0, 0.0))
    c2 = np.cumsum(np.insert(v * v, 0, 0.0))
    m = (c1[w:] - c1[:-w]) / w
    q = (c2[w:] - c2[:-w]) / w
    return np.sqrt(np.maximum(q - m * m, 0.0))


def _regime_coverage(r: np.ndarray, s: np.ndarray, w: int = 168) -> float:
    rs, ss = _rolling_std(r, w), _rolling_std(s, w)
    if rs.size < 10 * w // w + 10 or ss.size == 0:
        return 0.0
    edges = np.quantile(rs, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    occ_r = np.histogram(rs, edges)[0] / rs.size
    occ_s = np.histogram(ss, edges)[0] / ss.size
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(occ_r > 0, occ_s / occ_r, np.inf)
    return float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))


# ------------------------------------------------------------------ generator diagnostics
def generator_diagnostics(real: np.ndarray, synth: np.ndarray, obs: np.ndarray, prefix_invariance: bool,
                          nlags: int = 48) -> Dict[str, float]:
    real = np.asarray(real, np.float64)
    synth = np.asarray(synth, np.float64)
    m = np.asarray(obs, bool) & np.isfinite(real) & np.isfinite(synth)
    r, s = real[m], synth[m]
    out = {"n_val_observed_rows": int(r.size), "prefix_invariance": bool(prefix_invariance)}
    if r.size < 64:
        out.update(acf_max_abs_diff_lags_1_48=np.nan, psd_log_ratio_rmse=np.nan, ks_marginal=np.nan,
                   tail_ratio_q99_abs=np.nan, kurtosis_ratio=np.nan, regime_coverage=0.0)
        return out
    mu, sd = r.mean(), r.std()
    sd = sd if sd > 0 else 1.0
    rz, sz = (r - mu) / sd, (s - mu) / sd
    out["acf_max_abs_diff_lags_1_48"] = float(np.max(np.abs(_acf(rz, nlags) - _acf(sz, nlags))))
    nper = int(min(256, 2 ** int(np.floor(np.log2(r.size)))))
    _, pr = signal.welch(rz, nperseg=nper)
    _, ps = signal.welch(sz, nperseg=nper)
    out["psd_log_ratio_rmse"] = float(np.sqrt(np.mean(np.log10((ps + 1e-12) / (pr + 1e-12)) ** 2)))
    out["ks_marginal"] = float(stats.ks_2samp(rz, sz).statistic)
    qr, qs = np.quantile(np.abs(rz), 0.99), np.quantile(np.abs(sz), 0.99)
    out["tail_ratio_q99_abs"] = float(qs / qr) if qr > 0 else np.inf
    kr, ks = stats.kurtosis(rz, fisher=False), stats.kurtosis(sz, fisher=False)
    out["kurtosis_ratio"] = float(ks / kr) if kr > 0 else np.inf
    out["regime_coverage"] = _regime_coverage(rz, sz)
    return out


def _gate(name: str, spec: dict, value) -> bool:
    if "must_be" in spec:
        return bool(value) == bool(spec["must_be"])
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    if "min" in spec and value < spec["min"]:
        return False
    if "max" in spec and value > spec["max"]:
        return False
    return True


def decide_generator(metrics: dict, rule: dict) -> Tuple[str, List[str]]:
    failing = []
    for name, spec in rule["generator"].items():
        key = "n_val_observed_rows" if name == "min_val_observed_rows" else name
        if not _gate(name, spec, metrics.get(key)):
            failing.append(name)
    return ("CALIBRATED" if not failing else "NOT_CALIBRATED"), failing


# ------------------------------------------------------------------ knockoff exchangeability
def _corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(0)
    B = B - B.mean(0)
    sa = np.sqrt((A * A).sum(0))
    sb = np.sqrt((B * B).sum(0))
    sa[sa == 0] = np.inf
    sb[sb == 0] = np.inf
    return (A.T @ B) / np.outer(sa, sb)


def second_moment_gap(X: np.ndarray, Xk: np.ndarray) -> np.ndarray:
    """Per feature j: max over k != j of |corr(Xj,Xk) - corr(X~j,Xk)| and |corr(X~j,X~k) - corr(Xj,Xk)|."""
    p = X.shape[1]
    Cxx, Ckx, Ckk = _corr(X, X), _corr(Xk, X), _corr(Xk, Xk)
    off = ~np.eye(p, dtype=bool)
    g1 = np.where(off, np.abs(Cxx - Ckx), 0.0).max(1)
    g2 = np.where(off, np.abs(Ckk - Cxx), 0.0).max(1)
    gap = np.maximum(g1, g2)
    dead = (X.std(0) == 0) | (Xk.std(0) == 0)
    gap[dead] = 1.0
    return gap


def swap_classifier_auc(X: np.ndarray, Xk: np.ndarray, rng: np.random.Generator, max_rows: int = 20000) -> float:
    """AUC of a logistic classifier distinguishing [X,X~] from the same rows with a random column subset swapped."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    n, p = X.shape
    rows = np.arange(n) if n <= max_rows else np.linspace(0, n - 1, max_rows).astype(int)
    A = np.hstack([X[rows], Xk[rows]])
    swap = rng.random(p) < 0.5
    B = A.copy()
    B[:, np.r_[swap, np.zeros(p, bool)]] = A[:, np.r_[np.zeros(p, bool), swap]]
    B[:, np.r_[np.zeros(p, bool), swap]] = A[:, np.r_[swap, np.zeros(p, bool)]]
    half = rows.size // 2
    mu, sd = A[:half].mean(0), A[:half].std(0)
    sd[sd == 0] = 1.0
    Xtr = np.vstack([A[:half], B[:half]])
    ytr = np.r_[np.zeros(half), np.ones(half)]
    Xte = np.vstack([A[half:], B[half:]])
    yte = np.r_[np.zeros(rows.size - half), np.ones(rows.size - half)]
    clf = LogisticRegression(max_iter=300, C=1.0).fit((Xtr - mu) / sd, ytr)
    return float(roc_auc_score(yte, clf.decision_function((Xte - mu) / sd)))


def conditional_independence_abs_z(y: np.ndarray, X: np.ndarray, Xk: np.ndarray) -> np.ndarray:
    """|z| of the partial correlation of y with X~_j given X_j (leak guard; y never enters the generator)."""
    y = np.asarray(y, np.float64)
    n = y.size
    out = np.zeros(X.shape[1])
    yc = y - y.mean()
    for j in range(X.shape[1]):
        xj = X[:, j] - X[:, j].mean()
        d = float(xj @ xj)
        if d <= 0:
            continue
        kj = Xk[:, j] - Xk[:, j].mean()
        ry = yc - xj * (float(xj @ yc) / d)
        rk = kj - xj * (float(xj @ kj) / d)
        den = np.sqrt(float(ry @ ry) * float(rk @ rk))
        if den <= 0:
            continue
        r = np.clip(float(ry @ rk) / den, -0.999999, 0.999999)
        out[j] = abs(np.arctanh(r)) * np.sqrt(max(n - 4, 1))
    return out
