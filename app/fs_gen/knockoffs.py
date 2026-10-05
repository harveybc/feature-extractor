"""Second-order Gaussian group knockoffs, Gaussian-copula marginal map and the knockoff+ filter.

Candes, Fan, Janson, Lv (2018) Model-X knockoffs; Dai and Barber (2016) group equicorrelated S;
Barber and Candes (2015) knockoff+ threshold. In-repo implementation (knockpy is MIT but is not
installed in the worker environment; see fs_closure/fs_gen/CALIBRATION_RULE.json).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy import stats


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _inv_sqrt_psd(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(_sym(A))
    w = np.maximum(w, 1e-12)
    return (V / np.sqrt(w)) @ V.T


def group_equicorrelated_S(Sigma: np.ndarray, groups: Sequence[Sequence[int]], shrink: float = 1e-6) -> np.ndarray:
    """S = gamma * blockdiag(Sigma_gg), gamma = min(1, 2 lambda_min(D^-1/2 Sigma D^-1/2)) (Dai & Barber 2016)."""
    Sigma = _sym(np.asarray(Sigma, np.float64))
    p = Sigma.shape[0]
    D = np.zeros_like(Sigma)
    Dih = np.zeros_like(Sigma)
    for g in groups:
        g = np.asarray(g, int)
        blk = Sigma[np.ix_(g, g)]
        D[np.ix_(g, g)] = blk
        Dih[np.ix_(g, g)] = _inv_sqrt_psd(blk)
    M = _sym(Dih @ Sigma @ Dih)
    lam = float(np.linalg.eigvalsh(M).min())
    gamma = min(1.0, 2.0 * max(lam, 0.0)) * (1.0 - shrink)
    return gamma * D


def gaussian_knockoffs(X: np.ndarray, Sigma: np.ndarray, S: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """X~ = X (I - Sigma^-1 S) + N(0, 2S - S Sigma^-1 S) for centered X."""
    Sigma = _sym(np.asarray(Sigma, np.float64))
    SinvS = np.linalg.solve(Sigma, S)
    mu = X - X @ SinvS
    V = _sym(2.0 * S - S @ SinvS)
    w, U = np.linalg.eigh(V)
    L = U * np.sqrt(np.maximum(w, 0.0))
    return mu + rng.standard_normal(X.shape) @ L.T


def copula_map(z: np.ndarray, emp: np.ndarray) -> np.ndarray:
    """Gaussian copula: Phi(z) -> empirical quantile of `emp` (preserves the empirical marginal law)."""
    emp = np.sort(np.asarray(emp, np.float64))
    n = emp.size
    u = np.clip(stats.norm.cdf(np.asarray(z, np.float64)), 0.5 / n, 1.0 - 0.5 / n)
    grid = (np.arange(n) + 0.5) / n
    return np.interp(u, grid, emp)


def group_lasso_diff_stat(X: np.ndarray, Xk: np.ndarray, y: np.ndarray, groups: Sequence[Sequence[int]],
                          seed: int = 0, alpha: Optional[float] = None, n_alphas: int = 20,
                          max_iter: int = 3000) -> np.ndarray:
    """W_g = sum_{j in g} |beta_j| - sum_{j in g} |beta~_j| from one Lasso on [X, X~].

    alpha from contiguous 5-fold LassoCV when not given; the CV error is invariant to swapping a
    feature with its knockoff, so the statistic keeps the required antisymmetry.
    """
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.model_selection import KFold

    Z = np.hstack([X, Xk]).astype(np.float64)
    mu, sd = Z.mean(0), Z.std(0)
    sd[sd == 0] = 1.0
    Z = (Z - mu) / sd
    yc = np.asarray(y, np.float64)
    yc = yc - yc.mean()
    if alpha is None:
        cv = KFold(n_splits=5, shuffle=False)
        model = LassoCV(cv=cv, n_alphas=n_alphas, max_iter=max_iter, random_state=seed, n_jobs=1).fit(Z, yc)
        coef = model.coef_
    else:
        coef = Lasso(alpha=alpha, max_iter=max_iter, random_state=seed).fit(Z, yc).coef_
    p = X.shape[1]
    b, bk = np.abs(coef[:p]), np.abs(coef[p:])
    return np.array([float(b[list(g)].sum() - bk[list(g)].sum()) for g in groups])


def knockoff_threshold(W: np.ndarray, q: float, offset: int = 1) -> float:
    """Knockoff+ (offset=1) threshold; inf when no threshold controls the FDR at q."""
    W = np.asarray(W, np.float64)
    ts = np.sort(np.unique(np.abs(W[W != 0])))
    for t in ts:
        num = offset + int(np.sum(W <= -t))
        den = max(1, int(np.sum(W >= t)))
        if num / den <= q:
            return float(t)
    return float("inf")


def groups_from_clusters(feature_ids: List[str], clusters: Sequence[Sequence[str]]) -> List[List[int]]:
    """Map lane B dependence clusters (feature-id lists) to index groups over `feature_ids`; singletons otherwise."""
    pos = {f: i for i, f in enumerate(feature_ids)}
    seen = set()
    groups: List[List[int]] = []
    for c in clusters:
        idx = sorted(pos[f] for f in c if f in pos and f not in seen)
        if idx:
            groups.append(idx)
            seen.update(feature_ids[i] for i in idx)
    for f, i in pos.items():
        if f not in seen:
            groups.append([i])
    return sorted(groups)
