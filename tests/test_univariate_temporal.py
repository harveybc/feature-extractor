"""Contract tests for the univariate temporal extractor (lane D, 2026-10-03).

Written before the implementation. Tiny synthetic data, CPU only.
"""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import pytest

from app import univariate_temporal as U

T = 24
C = 6
TINY = dict(window=T, calendar_dim=C, latent_dim=8, filters=4, kernel_size=3,
            dilations=(1, 2), decoder_filters=4)


def _series(n=200, seed=0, missing=0.1):
    rng = np.random.default_rng(seed)
    ts = (1_700_000_000 // 3600) * 3600 + 3600 * np.arange(n, dtype=np.int64)
    x = np.cumsum(rng.normal(size=n)).astype(np.float32)
    obs = rng.random(n) > missing
    return ts, x, obs


def _batch(n_windows=16, seed=0, window=T):
    ts, x, obs = _series(n=window + n_windows + 10, seed=seed)
    norm = U.Normalization.fit(x, obs, np.arange(len(x)))
    anchors = np.arange(window - 1, window - 1 + n_windows)
    return U.make_windows(ts, x, obs, anchors, window, norm)


def _cfg(**kw):
    d = dict(TINY)
    d.update(kw)
    return U.ArchConfig(**d)


# 1. future perturbation does not change past latents -------------------------
@pytest.mark.parametrize("family", ["random", "ae", "dae"])
def test_future_perturbation_does_not_change_past_latents(family):
    b = _batch()
    ext = U.make_extractor(family, _cfg(), seed=1)
    if U.FAMILIES[family].trainable:
        ext.fit(b, b, U.EarlyStopConfig(max_epochs=1, patience=1))
    z = ext.encode(b)
    t0 = 13
    p = b.copy_arrays()
    rng = np.random.default_rng(9)
    p.signal[:, t0:, :] += rng.normal(size=p.signal[:, t0:, :].shape).astype(np.float32) * 5
    p.observed_mask[:, t0:, :] = 1.0 - p.observed_mask[:, t0:, :]
    p.signal[:, t0:, :] *= p.observed_mask[:, t0:, :]
    p.delta_time[:, t0:, :] += 3.0
    p.calendar[:, t0:, :] = rng.normal(size=p.calendar[:, t0:, :].shape).astype(np.float32)
    zp = ext.encode(p)
    np.testing.assert_allclose(zp[:, :t0], z[:, :t0], atol=1e-5)
    assert not np.allclose(zp[:, t0:], z[:, t0:])
    if U.FAMILIES[family].has_decoder:
        np.testing.assert_allclose(ext.reconstruct(p)[:, :t0], ext.reconstruct(b)[:, :t0], atol=1e-5)


# 2. target forbidden in the operational encoder -------------------------------
def test_target_forbidden_in_operational_encoder(tmp_path):
    ext = U.make_extractor("ae", _cfg(), seed=0)
    assert sorted(i.name for i in ext.encoder.inputs) == sorted(U.INPUT_NAMES)
    for bad in ("Y_s", "y", "Y_b", "target_h6", "label", "future_return"):
        with pytest.raises(U.TargetLeakError):
            U.check_no_target({bad: np.zeros(1)})
    b = _batch()
    with pytest.raises(U.TargetLeakError):
        ext.encode_inputs(dict(b.as_inputs(), Y_s=np.zeros((len(b), 6), np.float32)))
    # typed NPZ route through the single adapter line
    from app import npz_encoder_adapter as A
    p = str(tmp_path / "w.npz")
    np.savez(p, split=np.array("train"), row_ids=b.row_ids, Y_l=np.zeros(len(b)), **b.as_inputs())
    with pytest.raises(U.TargetLeakError):
        A.load_univariate_temporal_npz(p)
    q = str(tmp_path / "ok.npz")
    np.savez(q, split=np.array("train"), row_ids=b.row_ids, **b.as_inputs())
    lb = A.load_univariate_temporal_npz(q)
    assert lb.signal.shape == b.signal.shape


# 3. time grid preserved -------------------------------------------------------
@pytest.mark.parametrize("window", [24, 48])
@pytest.mark.parametrize("family", ["identity", "random", "ae", "dae"])
def test_time_grid_preserved(window, family):
    b = _batch(window=window)
    ext = U.make_extractor(family, _cfg(window=window), seed=0)
    z = ext.encode(b)
    assert z.shape[:2] == (len(b), window)
    assert z.shape[2] == ext.latent_dim
    U.assert_temporal_output(z, len(b), window)


def test_pooled_output_violates_temporal_contract():
    with pytest.raises(U.TemporalContractError):
        U.assert_temporal_output(np.zeros((4, 8), np.float32), 4, T)
    with pytest.raises(U.TemporalContractError):
        U.assert_temporal_output(np.zeros((4, T // 2, 8), np.float32), 4, T)
    import keras
    inp = keras.Input((T, 1))
    pooled = keras.Model(inp, keras.layers.GlobalAveragePooling1D()(inp))
    with pytest.raises(U.TemporalContractError):
        U.assert_temporal_encoder(pooled, T)


# 4. calendar known at t only --------------------------------------------------
def test_calendar_known_at_t_only():
    ts, _, _ = _series(n=100)
    cal = U.calendar_features(ts)
    assert cal.shape == (100, 6) and cal.dtype == np.float32
    ts2 = ts.copy()
    ts2[50:] += 86400 * 37 + 3600 * 5  # rewrite the future
    np.testing.assert_array_equal(U.calendar_features(ts2)[:50], cal[:50])
    midnight_monday_jan1 = np.array([1704067200], dtype=np.int64)  # 2024-01-01 00:00 UTC, Monday
    c = U.calendar_features(midnight_monday_jan1)[0]
    np.testing.assert_allclose(c[[0, 1, 2, 3]], [0.0, 1.0, 0.0, 1.0], atol=1e-6)
    vals = np.ones(100, np.float32)
    pub_ok = ts - 1
    out = U.known_calendar_columns(ts, {"session": (vals, pub_ok)})
    assert out.shape == (100, 1)
    pub_bad = pub_ok.copy()
    pub_bad[40] = ts[40] + 1
    with pytest.raises(U.CalendarLeakError):
        U.known_calendar_columns(ts, {"holiday": (vals, pub_bad)})


# 5. raw / random / trained share interface and shape ---------------------------
def test_controls_share_interface_and_shape():
    b = _batch()
    exts = {f: U.make_extractor(f, _cfg(), seed=0) for f in ("identity", "random", "ae", "dae")}
    exts["ae"].fit(b, b, U.EarlyStopConfig(max_epochs=1, patience=1))
    shapes = {f: e.encode(b).shape for f, e in exts.items()}
    assert shapes["random"] == shapes["ae"] == shapes["dae"] == (len(b), T, 8)
    assert shapes["identity"] == (len(b), T, 3)  # declared, not padded to D
    assert exts["random"].architecture_id == exts["ae"].architecture_id == exts["dae"].architecture_id
    assert exts["random"].encoder.count_params() == exts["ae"].encoder.count_params()
    assert exts["dae"].corruption == {"type": "gaussian_observed", "sigma": 0.1}
    assert exts["ae"].corruption is None
    with pytest.raises(U.ContractError):
        U.make_extractor("vae_of_the_week", _cfg(), seed=0)


LANE_F_TINY = {"d_model": 8, "n_blocks": 1, "n_heads": 2, "max_lag": T, "pairs_per_epoch": 32}


@pytest.mark.parametrize("family", ["masked_temporal_ae", "past_to_current_siamese"])
def test_lane_f_slots_wired_through_the_same_interface(family, tmp_path):
    """The two reserved slots run lane F's single implementation behind lane D's interface."""
    assert U.FAMILIES[family].status == U.IMPLEMENTED_LANE_F
    b, v = _batch(n_windows=40), _batch(n_windows=40, seed=5)
    ext = U.make_extractor(family, _cfg(), seed=0, alt=LANE_F_TINY)
    assert sorted(i.name for i in ext.encoder.inputs) == sorted(U.INPUT_NAMES)
    rep = ext.fit(b, v, U.EarlyStopConfig(max_epochs=2, patience=1))
    assert rep["restored_best_checkpoint"] and rep["updates"] > 0
    z = ext.encode(b)
    U.assert_temporal_output(z, len(b), T)
    assert ext.reconstruct(b) is None  # NOT_APPLICABLE, not a failure (FS17)
    p = b.copy_arrays()
    p.signal[:, 13:, :] += 5.0 * p.observed_mask[:, 13:, :]
    np.testing.assert_allclose(ext.encode(p)[:, :13], z[:, :13], atol=1e-5)
    with pytest.raises(U.TargetLeakError):
        ext.encode_inputs(dict(b.as_inputs(), Y_s=np.zeros((len(b), 6), np.float32)))
    scope = U.TrainScope("TRAIN_ONLY", "train", "f0", int(b.anchor_ts[0]), int(b.anchor_ts[-1]),
                         "d" * 64, U.row_ids_sha256(b.row_ids))
    man = ext.export_donor(str(tmp_path / "d"), scope, feature_id="feat_a")
    assert man["architecture_id"] == U.LANE_F_ARCHITECTURE_IDS[family]
    enc2, _ = U.load_donor(str(tmp_path / "d"), "R2")
    np.testing.assert_allclose(enc2.predict(b.as_inputs(), verbose=0), z, atol=1e-6)
    enc0, rec0 = U.load_donor(str(tmp_path / "d"), "R0", seed=9)
    assert rec0["initial_weights_sha256"] != man["weights_sha256"]
    if family == "past_to_current_siamese":
        with pytest.raises(U.ContractError):  # subsampled windows are not a contiguous series
            ext.fit(b.subset(np.arange(0, 40, 2)), v, U.EarlyStopConfig(max_epochs=1, patience=1))


# 6. TRAIN-only folds ----------------------------------------------------------
def test_train_only_folds():
    ts, x, obs = _series(n=400)
    good = U.FoldSpec("f0", "train", fit=(int(ts[30]), int(ts[200])), val=(int(ts[200 + T + 5]), int(ts[300])))
    U.validate_fold(good, ts, train_end_ts=int(ts[399]), window=T)
    with pytest.raises(U.FoldScopeError):
        U.validate_fold(U.FoldSpec("f1", "validation", good.fit, good.val), ts, int(ts[399]), T)
    with pytest.raises(U.FoldScopeError):  # beyond TRAIN end
        U.validate_fold(good, ts, train_end_ts=int(ts[250]), window=T)
    with pytest.raises(U.FoldScopeError):  # val windows overlap fit anchors
        U.validate_fold(U.FoldSpec("f2", "train", good.fit, (int(ts[205]), int(ts[300]))), ts, int(ts[399]), T)
    fit_idx = U.fold_anchor_indices(ts, good.fit)
    n1 = U.Normalization.fit(x, obs, U.covered_indices(fit_idx, T))
    x2 = x.copy()
    x2[260:] += 1000.0  # outside every fit window
    n2 = U.Normalization.fit(x2, obs, U.covered_indices(fit_idx, T))
    assert (n1.mean, n1.std) == (n2.mean, n2.std)


# 7. save/load parity and R0/R1/R2 ---------------------------------------------
def test_save_load_parity_and_regimes(tmp_path):
    b = _batch()
    ext = U.make_extractor("ae", _cfg(), seed=2)
    ext.fit(b, b, U.EarlyStopConfig(max_epochs=2, patience=1))
    scope = U.TrainScope(kind="TRAIN_ONLY", split="train", fold_id="f0",
                         fit_first_ts=int(b.anchor_ts[0]), fit_last_ts=int(b.anchor_ts[-1]),
                         input_sha256="a" * 64, train_row_ids_sha256=U.row_ids_sha256(b.row_ids))
    man = ext.export_donor(str(tmp_path / "d"), scope, feature_id="feat_a")
    assert not any("decoder" in f for f in os.listdir(tmp_path / "d"))
    disk = json.load(open(tmp_path / "d" / "donor_manifest.json"))
    for k in ("input_sha256", "train_row_ids_sha256", "weights_sha256", "encoder_sha256",
              "architecture_id", "seed", "train_scope", "regimes"):
        assert k in disk
    assert disk["regimes"] == ["R0", "R1", "R2"]
    z = ext.encode(b)
    enc2, rec2 = U.load_donor(str(tmp_path / "d"), "R2")
    np.testing.assert_array_equal(enc2.predict(b.as_inputs(), verbose=0), z)
    assert rec2["initial_weights_sha256"] == man["weights_sha256"]
    enc1, _ = U.load_donor(str(tmp_path / "d"), "R1")
    assert not enc1.trainable
    before = U.weights_sha256(enc1)
    U.one_update_step(enc1, b)
    assert U.weights_sha256(enc1) == before
    U.one_update_step(enc2, b)
    assert U.weights_sha256(enc2) != man["weights_sha256"]
    enc0, rec0 = U.load_donor(str(tmp_path / "d"), "R0", seed=5)
    assert enc0.trainable and rec0["initial_weights_sha256"] != man["weights_sha256"]
    assert [w.shape for w in enc0.get_weights()] == [w.shape for w in enc2.get_weights()]


# 8. early stopping restores the best checkpoint -------------------------------
def test_early_stop_restores_best():
    b = _batch(n_windows=24)
    v = _batch(n_windows=12, seed=3)
    ext = U.make_extractor("ae", _cfg(), seed=4, learning_rate=0.3)
    rep = ext.fit(b, v, U.EarlyStopConfig(max_epochs=12, patience=2, min_delta=0.0))
    assert rep["stop_reason"] in ("patience", "max_epochs")
    assert rep["epochs_run"] >= rep["best_epoch"] + 1
    best = rep["best_epoch"]
    assert rep["weights_sha256_by_epoch"][best] == U.weights_sha256(ext.training_model)
    assert ext.validation_loss(v) == pytest.approx(rep["best_val_loss"], rel=1e-5, abs=1e-7)
    assert rep["updates"] > 0 and rep["updates_at_best"] <= rep["updates"]


# 9. donor digest mismatch refused ---------------------------------------------
def test_donor_identity_mismatch_refused(tmp_path):
    b = _batch()
    ext = U.make_extractor("random", _cfg(), seed=0)
    scope = U.TrainScope(kind="TRAIN_ONLY", split="train", fold_id="f0", fit_first_ts=0, fit_last_ts=1,
                         input_sha256="b" * 64, train_row_ids_sha256=U.row_ids_sha256(b.row_ids))
    d = str(tmp_path / "d")
    ext.export_donor(d, scope, feature_id="feat_a")
    U.load_donor(d, "R1", expected={"input_sha256": "b" * 64, "feature_id": "feat_a", "window": T})
    for exp in ({"input_sha256": "c" * 64}, {"feature_id": "feat_b"}, {"window": 48},
                {"architecture_id": "other"}):
        with pytest.raises(U.DonorIdentityError):
            U.load_donor(d, "R1", expected=exp)
    with pytest.raises(U.DonorIdentityError):  # foreign corpus is never TRAIN_ONLY for this batch
        U.assert_train_only(json.load(open(os.path.join(d, "donor_manifest.json"))), "c" * 64)
    m = json.load(open(os.path.join(d, "donor_manifest.json")))
    m["weights_sha256"] = "0" * 64
    json.dump(m, open(os.path.join(d, "donor_manifest.json"), "w"))
    with pytest.raises(U.DonorIdentityError):
        U.load_donor(d, "R1")
    ext.export_donor(d, scope, feature_id="feat_a")
    with open(os.path.join(d, "encoder.keras"), "ab") as f:
        f.write(b"x")
    with pytest.raises(U.DonorIdentityError):
        U.load_donor(d, "R2")
    with pytest.raises(U.DonorIdentityError):
        U.load_donor(str(tmp_path / "missing"), "R1")
    with pytest.raises(U.DonorIdentityError):
        U.TrainScope(kind="TRAIN_ONLY", split="validation", fold_id="f0", fit_first_ts=0, fit_last_ts=1,
                     input_sha256="b" * 64, train_row_ids_sha256="x").validate()
