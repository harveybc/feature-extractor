"""Per feature/fold metrics for univariate temporal extractors (lane D, 2026-10-03).

Reconstruction is a diagnostic, never a selection criterion (FS05/FS06): a family
without a decoder reports NOT_APPLICABLE, not a failure (FS17). Probes are equal
for raw / random / trained representations and use the target ONLY as probe
supervision (FS03). Every regression row carries the same-row naive and skill.
Values are reported unrounded.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from app.univariate_temporal import check_no_target

DEFAULT_PROBE_LAGS = (0, 1, 2, 23)


# --------------------------------------------------------------------------- reconstruction
def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    den = float(np.dot(x, x))
    if den <= 0:
        return np.zeros(max_lag)
    return np.array([np.dot(x[:-k], x[k:]) / den for k in range(1, max_lag + 1)])


def _log_psd(x: np.ndarray) -> np.ndarray:
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    return np.log(p[1:] + 1e-12)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int) -> float:
    n = len(a)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        for j in range(lo, hi + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, n] / n)


def reconstruction_metrics(x_norm, xhat_norm, mask, mean: float, std: float, extreme_threshold_norm: float,
                           max_lag: int = 24, dtw_windows: int = 16, dtw_band: Optional[int] = None) -> dict:
    if xhat_norm is None:
        return {"status": "NOT_APPLICABLE", "reason": "family/export has no decoder; reconstruction is not defined"}
    x = np.asarray(x_norm, np.float64)[..., 0]
    xh = np.asarray(xhat_norm, np.float64)[..., 0]
    m = np.asarray(mask)[..., 0] > 0
    if not m.any():
        return {"status": "FAILED", "reason": "no observed points"}
    err = (xh - x)[m]
    out = {"status": "MEASURED", "n_points": int(m.sum()),
           "mae_norm": float(np.mean(np.abs(err))), "mse_norm": float(np.mean(err ** 2)),
           "mae_orig": float(np.mean(np.abs(err * std))), "mse_orig": float(np.mean((err * std) ** 2))}
    const_mae, const_mse = float(np.mean(np.abs(x[m]))), float(np.mean(x[m] ** 2))  # TRAIN-mean constant == 0
    out["train_constant_mae_norm"], out["train_constant_mse_norm"] = const_mae, const_mse
    out["mae_rel_train_constant"] = out["mae_norm"] / const_mae if const_mae > 0 else None
    out["mse_rel_train_constant"] = out["mse_norm"] / const_mse if const_mse > 0 else None
    ext = m & (np.abs(x) > extreme_threshold_norm)
    out["n_extremes"] = int(ext.sum())
    out["extreme_threshold_norm"] = float(extreme_threshold_norm)
    out["mae_extremes"] = float(np.mean(np.abs((xh - x)[ext]))) if ext.any() else None
    T = x.shape[1]
    lag = max(1, min(max_lag, T - 1))
    xf, xhf = np.where(m, x, 0.0), np.where(m, xh, 0.0)  # missing compared as 0 on both sides
    out["acf_lags"] = lag
    out["acf_l1"] = float(np.mean([np.mean(np.abs(_acf(a, lag) - _acf(b, lag))) for a, b in zip(xf, xhf)]))
    out["log_psd_l1"] = float(np.mean([np.mean(np.abs(_log_psd(a) - _log_psd(b))) for a, b in zip(xf, xhf)]))
    band = dtw_band if dtw_band is not None else max(1, T // 10)
    k = min(dtw_windows, x.shape[0])
    out["dtw_band"], out["dtw_windows"] = band, k
    out["dtw_mean"] = float(np.mean([dtw_distance(xf[i], xhf[i], band) for i in range(k)]))
    vals = [v for v in out.values() if isinstance(v, float)]
    if not np.isfinite(vals).all():
        out["status"], out["reason"] = "FAILED", "non-finite metric"
    return out


# --------------------------------------------------------------------------- latent
def effective_dimension(z) -> dict:
    a = np.asarray(z, np.float64).reshape(-1, np.shape(z)[-1])
    a = a - a.mean(0)
    ev = np.clip(np.linalg.eigvalsh(a.T @ a / max(len(a) - 1, 1)), 0, None)[::-1]
    s = ev.sum()
    if s <= 0:
        return {"participation_ratio": 0.0, "n_components_95": 0, "dim": int(a.shape[1])}
    return {"participation_ratio": float(s ** 2 / np.sum(ev ** 2)),
            "n_components_95": int(np.searchsorted(np.cumsum(ev) / s, 0.95) + 1), "dim": int(a.shape[1])}


def linear_cka(a, b) -> float:
    a = np.asarray(a, np.float64).reshape(len(a), -1)
    b = np.asarray(b, np.float64).reshape(len(b), -1)
    a, b = a - a.mean(0), b - b.mean(0)
    num = np.linalg.norm(a.T @ b) ** 2
    den = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    return float(num / den) if den > 0 else 0.0


def stability_across_folds(latents_by_fold: Dict[str, np.ndarray]) -> dict:
    keys = sorted(latents_by_fold)
    if len(keys) < 2:
        return {"status": "NOT_APPLICABLE", "reason": "needs >= 2 folds"}
    vals = [linear_cka(latents_by_fold[i].reshape(-1, latents_by_fold[i].shape[-1]),
                       latents_by_fold[j].reshape(-1, latents_by_fold[j].shape[-1]))
            for n, i in enumerate(keys) for j in keys[n + 1:]]
    return {"status": "MEASURED", "metric": "linear_cka_on_reference_rows", "pairs": len(vals),
            "mean": float(np.mean(vals)), "min": float(np.min(vals))}


# --------------------------------------------------------------------------- probes
def probe_features(z, calendar, lags: Sequence[int] = DEFAULT_PROBE_LAGS) -> np.ndarray:
    z = np.asarray(z, np.float64)
    T = z.shape[1]
    parts = [z[:, T - 1 - l, :] for l in lags if l < T]
    parts.append(np.asarray(calendar, np.float64)[:, -1, :])
    return np.concatenate(parts, axis=1)


def _ridge(xf, yf, xv, alpha):
    xa = np.c_[xf, np.ones(len(xf))]
    reg = alpha * np.eye(xa.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xa.T @ xa + reg, xa.T @ yf)
    return np.c_[xv, np.ones(len(xv))] @ w


def _softmax_probe(xf, yf, xv, classes, iters=300, lr=0.5, l2=1e-3):
    k = len(classes)
    yi = np.searchsorted(classes, yf)
    onehot = np.eye(k)[yi]
    xa, xva = np.c_[xf, np.ones(len(xf))], np.c_[xv, np.ones(len(xv))]
    w = np.zeros((xa.shape[1], k))
    for _ in range(iters):
        logits = xa @ w
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        w -= lr * (xa.T @ (p - onehot) / len(xa) + l2 * w)
    logits = xva @ w
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def equal_probes(reps: Dict[str, np.ndarray], calendar, targets: Dict[str, np.ndarray], fit_idx, val_idx,
                 ridge_alpha: float = 1.0, lags: Sequence[int] = DEFAULT_PROBE_LAGS) -> list:
    """Same probe, same rows, same budget for every representation. Integer targets -> classification."""
    check_no_target(reps)  # a representation is never a target
    fit_idx, val_idx = np.asarray(fit_idx), np.asarray(val_idx)
    feats = {}
    for name, z in reps.items():
        f = probe_features(z, calendar, lags)
        mu, sd = f[fit_idx].mean(0), f[fit_idx].std(0)
        sd[sd < 1e-12] = 1.0
        feats[name] = (f - mu) / sd
    rows = []
    for tname, y in targets.items():
        y = np.asarray(y)
        y2 = y[:, None] if y.ndim == 1 else y
        is_cls = np.issubdtype(y.dtype, np.integer)
        for h in range(y2.shape[1]):
            col = y2[:, h]
            ok = (col >= 0) if is_cls else np.isfinite(col)
            fi, vi = fit_idx[ok[fit_idx]], val_idx[ok[val_idx]]
            base = {"target": tname, "horizon_index": h, "n_fit": int(fi.size), "n_val": int(vi.size),
                    "probe_lags": [int(l) for l in lags], "kind": "classification" if is_cls else "regression"}
            if fi.size < 2 or vi.size < 1:
                for name in feats:
                    rows.append(dict(base, representation=name, status="FAILED", reason="no supported rows"))
                continue
            if is_cls:
                classes = np.unique(col[fi])
                prior = np.array([(col[fi] == c).mean() for c in classes])
                yv = col[vi]
                known = np.isin(yv, classes)
                onehot = np.zeros((vi.size, len(classes)))
                onehot[known, np.searchsorted(classes, yv[known])] = 1.0
                pp = np.clip(np.tile(prior, (vi.size, 1)), 1e-12, 1)
                prior_ll = float(-np.mean(np.log(np.sum(pp * onehot, 1).clip(1e-12))))
                prior_brier = float(np.mean(np.sum((pp - onehot) ** 2, 1)))
                for name, f in feats.items():
                    p = np.clip(_softmax_probe(f[fi], col[fi], f[vi], classes), 1e-12, 1)
                    ll = float(-np.mean(np.log(np.sum(p * onehot, 1).clip(1e-12))))
                    rows.append(dict(base, representation=name, status="MEASURED", probe="softmax_l2",
                                     log_loss=ll, brier=float(np.mean(np.sum((p - onehot) ** 2, 1))),
                                     prior_log_loss=prior_ll, prior_brier=prior_brier,
                                     skill_log_loss_vs_prior=1 - ll / prior_ll if prior_ll > 0 else None,
                                     naive_n_val=int(vi.size)))
            else:
                yf, yv = col[fi].astype(np.float64), col[vi].astype(np.float64)
                z_mae, z_mse = float(np.mean(np.abs(yv))), float(np.mean(yv ** 2))
                tm_mae = float(np.mean(np.abs(yv - yf.mean())))
                for name, f in feats.items():
                    pred = _ridge(f[fi], yf, f[vi], ridge_alpha)
                    mae, mse = float(np.mean(np.abs(pred - yv))), float(np.mean((pred - yv) ** 2))
                    rows.append(dict(base, representation=name, status="MEASURED", probe="ridge",
                                     ridge_alpha=ridge_alpha, mae=mae, mse=mse, naive_n_val=int(vi.size),
                                     naive_zero_mae=z_mae, naive_zero_mse=z_mse, naive_train_mean_mae=tm_mae,
                                     skill_vs_zero=1 - mae / z_mae if z_mae > 0 else None,
                                     skill_vs_train_mean=1 - mae / tm_mae if tm_mae > 0 else None))
    return rows


def probe_deltas(rows: list, raw: str = "raw", random: str = "random", trained: str = "trained") -> dict:
    """Delta_probe = L(random) - L(trained); preservation = L(raw) - L(trained). Positive favours trained."""
    loss, kind = {}, {}
    for r in rows:
        if r.get("status") != "MEASURED":
            continue
        k = (r["target"], r["horizon_index"])
        kind[k] = "mae" if r["kind"] == "regression" else "log_loss"
        loss.setdefault(k, {})[r["representation"]] = r[kind[k]]
    out = {}
    for k, d in loss.items():
        if trained not in d:
            continue
        e = {"trained": trained, "loss": kind[k], "loss_trained": d[trained]}
        if random in d:
            e["delta_probe_random_minus_trained"] = d[random] - d[trained]
        if raw in d:
            e["preservation_raw_minus_trained"] = d[raw] - d[trained]
        out[k] = e
    return out
