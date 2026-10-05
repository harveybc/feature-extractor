"""Tests for lane FS-GEN: conditional temporal generator, calibration gates and group knockoffs.

Written before the implementation (red first). Tiny synthetic data, CPU only, one seed.
"""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import pytest

from app import univariate_temporal as U
from app.fs_gen import calibration as C
from app.fs_gen import contracts as K
from app.fs_gen import generator as G
from app.fs_gen import knockoffs as KO
from app.fs_gen import pipeline as P


def _grid(n, start=1_700_000_000):
    return (start // 3600) * 3600 + 3600 * np.arange(n, dtype=np.int64)


def _ar_series(n, seed, phi=0.8, missing=0.05, heavy=False):
    rng = np.random.default_rng(seed)
    ts = _grid(n)
    hour = (ts // 3600) % 24
    e = rng.standard_t(2.2, size=n) if heavy else rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    x = x + 0.3 * np.sin(2 * np.pi * hour / 24)
    obs = rng.random(n) > missing
    x = x.astype(np.float32)
    x[~obs] = np.nan
    return ts, x, obs


# 1. contracts ----------------------------------------------------------------
def test_target_as_generator_condition_is_refused():
    with pytest.raises(U.TargetLeakError):
        K.check_generator_conditions({"signal": 1, "observed_mask": 1, "delta_time": 1, "calendar": 1, "Y_s": 1})
    with pytest.raises(U.TargetLeakError):
        K.check_generator_conditions({"signal": 1, "observed_mask": 1, "delta_time": 1, "calendar": 1, "target_h6": 1})


def test_economic_calendar_is_refused_as_condition():
    with pytest.raises(U.EconomicCalendarInputError):
        K.check_calendar_columns(["session_london", "econ_nfp_surprise"])
    with pytest.raises(U.EconomicCalendarInputError):
        K.check_calendar_columns(["cpi_release"])
    K.check_calendar_columns(["session_london", "holiday_us"])  # allowed


def test_rows_after_train_end_are_refused():
    ts = _grid(100)
    with pytest.raises(U.FoldScopeError):
        K.assert_train_only(ts, int(ts[50]))
    K.assert_train_only(ts, int(ts[-1]))


def test_contract_is_declared_synthetic_offline():
    assert K.CONTRACT == "SYNTHETIC_OFFLINE"


# 2. generator ------------------------------------------------------------------
def test_generator_fits_and_synthesizes_with_mask_preserved():
    ts, x, obs = _ar_series(1500, seed=1)
    fit_idx = np.arange(30, 1000)
    g = G.ConditionalARGenerator(order=6, seed=0).fit(ts, x, obs, fit_idx)
    s = g.synthesize(ts, x, obs, 1000, 1500)
    assert s.shape == (500,)
    assert np.array_equal(np.isnan(s), ~obs[1000:1500])
    assert np.isfinite(s[obs[1000:1500]]).all()
    z, ok = g.innovations(ts, x, obs, np.arange(30, 1500))
    assert z.shape == ok.shape == (1470,)
    assert np.all(z[~ok] == 0.0)


def test_prefix_invariance_future_perturbation_changes_nothing():
    ts, x, obs = _ar_series(1500, seed=2)
    fit_idx = np.arange(30, 1000)
    g = G.ConditionalARGenerator(order=6, seed=0).fit(ts, x, obs, fit_idx)
    s0 = g.synthesize(ts, x, obs, 1000, 1500)
    xp = x.copy()
    xp[1000:] = np.where(np.isnan(xp[1000:]), np.nan, xp[1000:] + 50.0)
    s1 = g.synthesize(ts, xp, obs, 1000, 1500)
    assert np.array_equal(np.nan_to_num(s0), np.nan_to_num(s1))
    # innovations at t depend on rows <= t only
    idx = np.arange(30, 1500)
    z0, _ = g.innovations(ts, x, obs, idx)
    xq = x.copy()
    xq[1200:] = np.where(np.isnan(xq[1200:]), np.nan, xq[1200:] + 50.0)
    z1, _ = g.innovations(ts, xq, obs, idx)
    assert np.array_equal(z0[idx < 1200], z1[idx < 1200])
    assert not np.array_equal(z0[idx >= 1200], z1[idx >= 1200])
    # but the past before the synthesis start IS used
    xr = x.copy()
    xr[:1000] = np.where(np.isnan(xr[:1000]), np.nan, xr[:1000] * 3.0)
    s2 = G.ConditionalARGenerator(order=6, seed=0).fit(ts, xr, obs, fit_idx).synthesize(ts, xr, obs, 1000, 1500)
    assert not np.array_equal(np.nan_to_num(s0), np.nan_to_num(s2))


def test_generator_never_sees_target_argument():
    ts, x, obs = _ar_series(300, seed=3)
    g = G.ConditionalARGenerator(order=3, seed=0)
    with pytest.raises(TypeError):
        g.fit(ts, x, obs, np.arange(10, 200), target=np.zeros(300))  # noqa


# 3. calibration -----------------------------------------------------------------
def test_calibration_passes_on_well_specified_ar_and_fails_on_regime_switch():
    rule = C.default_rule()
    # validation size matches a real inner fold (~6k observed rows): the ACF gate is a fixed 0.10
    ts, x, obs = _ar_series(12000, seed=4)
    fit_idx = np.arange(30, 6000)
    g = G.ConditionalARGenerator(order=6, seed=0).fit(ts, x, obs, fit_idx)
    s = g.synthesize(ts, x, obs, 6000, 12000)
    pi = G.prefix_invariance_check(g, ts, x, obs, 6000, 12000)
    m = C.generator_diagnostics(x[6000:12000], s, obs[6000:12000], prefix_invariance=pi)
    state, failing = C.decide_generator(m, rule)
    assert state == "CALIBRATED", (failing, m)
    # regime switch in validation: variance x8 -> tails/regime coverage must fail closed
    xr = x.copy()
    xr[6000:] = xr[6000:] * 8.0
    s_r = g.synthesize(ts, xr, obs, 6000, 12000)
    m_r = C.generator_diagnostics(xr[6000:12000], s_r, obs[6000:12000], prefix_invariance=pi)
    state_r, failing_r = C.decide_generator(m_r, rule)
    assert state_r == "NOT_CALIBRATED" and failing_r


def test_too_few_validation_rows_is_not_calibrated():
    rule = C.default_rule()
    m = C.generator_diagnostics(np.zeros(10), np.zeros(10), np.ones(10, bool), prefix_invariance=True)
    state, failing = C.decide_generator(m, rule)
    assert state == "NOT_CALIBRATED" and "min_val_observed_rows" in failing


# 4. knockoffs ---------------------------------------------------------------------
def _planted(seed, n=600, p=40, k=8, rho=0.3, snr=3.0):
    rng = np.random.default_rng(seed)
    Sigma = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    true = rng.choice(p, k, replace=False)
    beta = np.zeros(p)
    beta[true] = rng.choice([-1, 1], k) * snr / np.sqrt(k)
    y = X @ beta + rng.normal(size=n)
    return X, y, true


def test_group_equicorrelated_S_is_psd_and_feasible():
    X, _, _ = _planted(0)
    Sigma = np.cov(X, rowvar=False)
    groups = [[0, 1, 2], [3], [4, 5]] + [[j] for j in range(6, X.shape[1])]
    S = KO.group_equicorrelated_S(Sigma, groups)
    assert np.all(np.linalg.eigvalsh(S) >= -1e-8)
    assert np.all(np.linalg.eigvalsh(2 * Sigma - S) >= -1e-8)


def test_second_order_knockoffs_match_moments_and_are_exchangeable():
    X, _, _ = _planted(1, n=4000)
    Xc = X - X.mean(0)
    Sigma = np.cov(Xc, rowvar=False)
    groups = [[j] for j in range(X.shape[1])]
    S = KO.group_equicorrelated_S(Sigma, groups)
    Xk = KO.gaussian_knockoffs(Xc, Sigma, S, np.random.default_rng(0))
    assert Xk.shape == Xc.shape
    gap = C.second_moment_gap(Xc, Xk)
    assert gap.shape == (X.shape[1],) and gap.max() < 0.08
    auc = C.swap_classifier_auc(Xc, Xk, np.random.default_rng(0))
    assert 0.42 < auc < 0.58


def test_knockoff_threshold_and_planted_truth_fdr_control():
    q = 0.2
    fdps, powers = [], []
    for seed in range(12):
        X, y, true = _planted(seed)
        Xc = X - X.mean(0)
        Sigma = np.cov(Xc, rowvar=False)
        groups = [[j] for j in range(X.shape[1])]
        S = KO.group_equicorrelated_S(Sigma, groups)
        Xk = KO.gaussian_knockoffs(Xc, Sigma, S, np.random.default_rng(seed))
        W = KO.group_lasso_diff_stat(Xc, Xk, y, groups, seed=seed)
        T = KO.knockoff_threshold(W, q, offset=1)
        sel = [g for g, w in zip(groups, W) if w >= T]
        sel_j = {j for g in sel for j in g}
        fdps.append(len(sel_j - set(true)) / max(1, len(sel_j)))
        powers.append(len(sel_j & set(true)) / len(true))
    assert np.mean(fdps) <= q + 0.1, fdps
    assert np.mean(powers) > 0.4, powers


def test_knockoff_threshold_is_inf_when_nothing_passes():
    W = np.array([-1.0, -0.5, 0.1, 0.2])
    T = KO.knockoff_threshold(W, 0.1, offset=1)
    assert T == np.inf


def test_copula_map_preserves_empirical_marginal():
    rng = np.random.default_rng(0)
    emp = rng.standard_t(3, size=5000)
    z = rng.normal(size=5000)
    mapped = KO.copula_map(z, emp)
    assert mapped.shape == z.shape
    q_emp = np.quantile(emp, [0.01, 0.5, 0.99])
    q_map = np.quantile(mapped, [0.01, 0.5, 0.99])
    assert np.allclose(q_emp, q_map, rtol=0.25, atol=0.15)


# 5. pipeline: NOT_CALIBRATED path and planted knockoff truth -------------------------
def _write_batch(tmp_path, n=5000, feats=("f_ok", "f_heavy"), seed=0):
    from app.univariate_temporal_pilot import write_synthetic_ps2_batch
    d = os.path.join(tmp_path, "batch_777")
    write_synthetic_ps2_batch(d, n=n, features=feats, seed=seed, window=24)
    # make f_heavy regime-switching and heavy tailed so it fails closed
    import hashlib
    series = dict(np.load(os.path.join(d, "series.npz"), allow_pickle=False))
    ts, xh, _ = _ar_series(n, seed=99, heavy=True)
    xh[n // 2:] *= 10.0
    series["x__f_heavy"] = xh
    np.savez(os.path.join(d, "series.npz"), **series)
    man = json.load(open(os.path.join(d, "batch_manifest.json")))
    man["series"]["sha256"] = hashlib.sha256(open(os.path.join(d, "series.npz"), "rb").read()).hexdigest()
    json.dump(man, open(os.path.join(d, "batch_manifest.json"), "w"))
    return d


def test_pipeline_emits_not_calibrated_and_progress(tmp_path):
    d = _write_batch(str(tmp_path))
    out = os.path.join(str(tmp_path), "out")
    res = P.run_batches([d], out_dir=out, groups_by_batch={}, seed=0, order=6, rule=C.default_rule(),
                        max_fit_rows=3000)
    assert os.path.isfile(os.path.join(out, "progress.json"))
    prog = json.load(open(os.path.join(out, "progress.json")))
    assert prog["cells_done"] == prog["cells_total"] and prog["state"] == "DONE"
    rows = res["fold_rows"]
    heavy = [r for r in rows if r["feature_id"] == "f_heavy"]
    assert heavy and all(r["generator_state"] == "NOT_CALIBRATED" for r in heavy)
    assert all(r["knockoff_state"] == "NOT_CALIBRATED" for r in heavy)
    assert all(r["knockoff_selected_cells"] == "" for r in heavy)
    summ = {r["feature_id"]: r for r in res["feature_rows"]}
    assert summ["f_heavy"]["calibration_state"] == "NOT_CALIBRATED"
    assert summ["f_heavy"]["knockoff_state"] == "NOT_CALIBRATED"
    assert set(summ) == {"f_ok", "f_heavy"}
    # resumable: a second run does no new cells
    res2 = P.run_batches([d], out_dir=out, groups_by_batch={}, seed=0, order=6, rule=C.default_rule(),
                         max_fit_rows=3000)
    assert res2["new_cells"] == 0


def test_denominator_rows_cover_missing_and_calendar(tmp_path):
    feature_rows = [{"feature_id": "f_ok", "calibration_state": "CALIBRATED", "knockoff_state": "RUN"}]
    denom = ["f_ok", "cal.hour_sin", "yh.missing.logret_1d"]
    out = P.denominator_rows(denom, feature_rows)
    by = {r["feature_id"]: r for r in out}
    assert len(out) == 3
    assert by["cal.hour_sin"]["calibration_state"] == "NOT_APPLICABLE"
    assert by["yh.missing.logret_1d"]["calibration_state"] == "NOT_EVALUATED"
    assert by["yh.missing.logret_1d"]["knockoff_state"] == "NOT_CALIBRATED"
    assert by["yh.missing.logret_1d"]["reason"] == "NO_SERIES_IN_PS2_BATCH"
