"""Conditional heteroskedastic AR generator for one feature on the regular hourly grid.

x_t = beta . [1, calendar_t, delta_t, lag_1..lag_p]  +  sigma_t * z_t
log(sigma_t^2) = gamma . [1, calendar_t, delta_t]
z_t ~ empirical standardized fit residuals (bootstrap)

Conditions are exactly the interface inputs (signal past, observed_mask, delta_time, calendar):
nothing from the future, no target, no economic calendar. Unobserved rows stay unobserved
(mask preserved; lag value 0 with the lag mask carried as information available at t).
Decision recorded in fs_closure/fs_gen/ACK.json: chosen over a CVAE because it fits the CPU
cap, gives an explicit conditional law (needed to argue knockoff exchangeability) and is
deterministic under one seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from app import univariate_temporal as U

MIN_FIT_ROWS = 300


def _lag_matrix(xn: np.ndarray, obs: np.ndarray, idx: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """The last `order` OBSERVED values strictly before each row of idx (information available at t).

    Gaps are skipped, not zero-filled: after a weekend the lags are Friday's values and delta_time
    carries the gap length. `allobs` is False where fewer than `order` observed values precede t.
    """
    obs_idx = np.nonzero(obs)[0]
    before = np.cumsum(obs) - obs.astype(np.int64)  # observed rows strictly before t
    nb = before[idx]
    vals = np.zeros((idx.size, order), np.float64)
    allobs = nb >= order
    for k in range(1, order + 1):
        pos = nb - k
        ok = pos >= 0
        src = obs_idx[np.where(ok, pos, 0)]
        vals[:, k - 1] = np.where(ok, xn[src], 0.0)
    return vals, allobs


@dataclass
class ConditionalARGenerator:
    order: int = 24
    seed: int = 0
    ridge: float = 1e-6
    mean: float = 0.0
    std: float = 1.0
    beta: Optional[np.ndarray] = None
    gamma: Optional[np.ndarray] = None
    z_emp: Optional[np.ndarray] = None
    n_fit: int = 0
    sigma_floor: float = 1e-3
    _cal_cache: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- conditioning
    def _conditions(self, ts: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        key = (ts.shape[0], int(ts[0]), int(ts[-1]), int(obs.sum()), hash(obs.tobytes()))
        if key not in self._cal_cache:
            self._cal_cache.clear()
            cal = U.calendar_features(ts).astype(np.float64)
            dt = U.delta_time_series(ts, obs).astype(np.float64)
            self._cal_cache[key] = (cal, dt)
        return self._cal_cache[key]

    def _mean_design(self, cal, dt, lags, idx):
        return np.column_stack([np.ones(idx.size), cal[idx], dt[idx], lags])

    def _scale_design(self, cal, dt, idx):
        return np.column_stack([np.ones(idx.size), cal[idx], dt[idx]])

    def _prep(self, ts, x, obs):
        ts = np.asarray(ts, np.int64)
        x = np.asarray(x, np.float64)
        obs = np.asarray(obs, bool) & np.isfinite(x)
        return ts, x, obs

    # ---------------------------------------------------------------- fit
    def fit(self, ts, x, obs, fit_idx) -> "ConditionalARGenerator":
        ts, x, obs = self._prep(ts, x, obs)
        fit_idx = np.asarray(fit_idx, np.int64)
        norm = U.Normalization.fit(x, obs, fit_idx)
        self.mean, self.std = norm.mean, norm.std
        xn = np.where(obs, (x - self.mean) / self.std, 0.0)
        cal, dt = self._conditions(ts, obs)
        lags, allobs = _lag_matrix(xn, obs, fit_idx, self.order)
        valid = obs[fit_idx] & allobs
        rows = fit_idx[valid]
        if rows.size < MIN_FIT_ROWS:
            raise U.ContractError(f"only {rows.size} valid fit rows (< {MIN_FIT_ROWS})")
        F = self._mean_design(cal, dt, lags[valid], rows)
        y = xn[rows]
        A = F.T @ F + self.ridge * np.eye(F.shape[1])
        self.beta = np.linalg.solve(A, F.T @ y)
        e = y - F @ self.beta
        G = self._scale_design(cal, dt, rows)
        le = np.log(e * e + 1e-12)
        Ag = G.T @ G + self.ridge * np.eye(G.shape[1])
        self.gamma = np.linalg.solve(Ag, G.T @ le)
        sig = self._sigma(G)
        z = e / sig
        z = z - z.mean()
        sd = z.std()
        self.z_emp = (z / sd if sd > 0 else z).astype(np.float64)
        self.gamma = self.gamma.copy()
        self.gamma[0] += 2.0 * np.log(sd if sd > 0 else 1.0)  # fold the re-standardization into the scale
        self.n_fit = int(rows.size)
        return self

    def _sigma(self, G):
        s = np.exp(0.5 * (G @ self.gamma))
        return np.maximum(s, self.sigma_floor)

    def _check_fitted(self):
        if self.beta is None:
            raise U.ContractError("generator not fitted")

    # ---------------------------------------------------------------- innovations (causal, row <= t)
    def innovations(self, ts, x, obs, idx) -> Tuple[np.ndarray, np.ndarray]:
        """Standardized innovations at rows idx; ok=False (and z=0) where x_t or a lag is unobserved."""
        self._check_fitted()
        ts, x, obs = self._prep(ts, x, obs)
        idx = np.asarray(idx, np.int64)
        xn = np.where(obs, (x - self.mean) / self.std, 0.0)
        cal, dt = self._conditions(ts, obs)
        lags, allobs = _lag_matrix(xn, obs, idx, self.order)
        ok = obs[idx] & allobs
        F = self._mean_design(cal, dt, lags, idx)
        G = self._scale_design(cal, dt, idx)
        z = (xn[idx] - F @ self.beta) / self._sigma(G)
        z = np.where(ok, z, 0.0)
        return z, ok

    def conditional_moments(self, ts, x, obs, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """mu_t, sigma_t (normalized units) given the REAL past, and the all-lags-observed flag."""
        self._check_fitted()
        ts, x, obs = self._prep(ts, x, obs)
        idx = np.asarray(idx, np.int64)
        xn = np.where(obs, (x - self.mean) / self.std, 0.0)
        cal, dt = self._conditions(ts, obs)
        lags, allobs = _lag_matrix(xn, obs, idx, self.order)
        F = self._mean_design(cal, dt, lags, idx)
        G = self._scale_design(cal, dt, idx)
        return F @ self.beta, self._sigma(G), obs[idx] & allobs

    # ---------------------------------------------------------------- free-running synthesis
    def synthesize(self, ts, x, obs, start: int, end: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Synthetic path over grid rows [start, end): real history before `start`, then free-running.

        Uses rows < start of x, and only the mask/calendar/delta (information available at t) inside
        the range. Returns the path in original units with NaN where unobserved.
        """
        self._check_fitted()
        ts, x, obs = self._prep(ts, x, obs)
        rng = np.random.default_rng(self.seed) if rng is None else rng
        w = np.where(obs, (x - self.mean) / self.std, 0.0)
        w[start:] = 0.0  # nothing real at or after start is used
        cal, dt = self._conditions(ts, obs)
        draws = rng.integers(0, self.z_emp.size, size=end - start)
        out = np.full(end - start, np.nan)
        p = self.order
        b0, bcal, bdt, blag = self.beta[0], self.beta[1:7], self.beta[7], self.beta[8:8 + p]
        hist = list(w[np.nonzero(obs[:start])[0]][-p:][::-1])  # last p observed values before start, lag1 first
        for i, t in enumerate(range(start, end)):
            if not obs[t]:
                continue
            lagv = np.array(hist[:p] + [0.0] * max(0, p - len(hist)))
            mu = b0 + bcal @ cal[t] + bdt * dt[t] + blag @ lagv
            g = np.concatenate([[1.0], cal[t], [dt[t]]])
            sig = max(np.exp(0.5 * (g @ self.gamma)), self.sigma_floor)
            w[t] = mu + sig * self.z_emp[draws[i]]
            hist.insert(0, w[t])
            if len(hist) > p:
                hist.pop()
            out[i] = w[t] * self.std + self.mean
        return out

    def to_dict(self) -> dict:
        return {"family": "conditional_heteroskedastic_ar", "order": self.order, "seed": self.seed,
                "n_fit": self.n_fit, "mean": self.mean, "std": self.std,
                "n_mean_params": int(self.beta.size), "n_scale_params": int(self.gamma.size)}


def prefix_invariance_check(g: ConditionalARGenerator, ts, x, obs, start: int, end: int) -> bool:
    """FS01-style: perturbing real values at or after `start` must not change the synthetic path."""
    s0 = g.synthesize(ts, x, obs, start, end)
    xp = np.asarray(x, np.float64).copy()
    xp[start:] = np.where(np.isnan(xp[start:]), np.nan, xp[start:] + 50.0 * (1.0 + np.abs(np.nanstd(xp))))
    s1 = g.synthesize(ts, xp, obs, start, end)
    return bool(np.array_equal(np.nan_to_num(s0), np.nan_to_num(s1)))
