"""Contract tests for the lane F alternative extractor families (MTAE, P2C)."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import pytest

from app import alt_extractor_families as F

T, C, D = 24, 6, 4


def _cfg():
    return F.EncoderConfig(window=T, calendar_dims=C, latent_dim=D, d_model=16, n_blocks=2, n_heads=2)


def _series(L=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L, dtype=np.float32)
    sig = (np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=L)).astype(np.float32)[:, None]
    cal = np.stack([np.sin(2 * np.pi * t / 24), np.cos(2 * np.pi * t / 24),
                    np.sin(2 * np.pi * t / 168), np.cos(2 * np.pi * t / 168),
                    np.sin(2 * np.pi * t / 8760), np.cos(2 * np.pi * t / 8760)], axis=1).astype(np.float32)
    return {"signal": sig, "observed_mask": np.ones((L, 1), np.float32),
            "delta_time": np.ones((L, 1), np.float32), "calendar": cal}


def _windows(n=32, seed=0):
    s = _series(seed=seed)
    return F.windows_from_series(s, T, np.arange(n) * 5)


@pytest.fixture(scope="module")
def mtae():
    w = _windows(64)
    v = _windows(16, seed=1)
    return F.fit_mtae(_cfg(), w, v, epochs=3, patience=2, batch_size=16, seed=0)


@pytest.fixture(scope="module")
def p2c():
    return F.fit_p2c(_cfg(), _series(seed=0), _series(seed=1), max_lag=3 * T, n_lineage=3,
                     pairs_per_epoch=48, epochs=3, patience=2, batch_size=16, seed=0)


@pytest.fixture(params=["mtae", "p2c"])
def encoder(request, mtae, p2c):
    return {"mtae": mtae, "p2c": p2c}[request.param][0]


def test_time_preserved(encoder):
    F.assert_temporal_contract(encoder)
    z = F.encode(encoder, _windows(8))
    assert z.shape == (8, T, D)
    assert np.all(np.isfinite(z))


@pytest.mark.parametrize("key", F.ENCODER_INPUTS)
def test_future_perturbation_does_not_reach_past_latent(encoder, key):
    x = _windows(4)
    z0 = F.encode(encoder, x)
    for t in (0, T // 2, T - 2):
        y = {k: v.copy() for k, v in x.items()}
        y[key][:, t + 1:, :] += 5.0
        z1 = F.encode(encoder, y)
        np.testing.assert_allclose(z1[:, : t + 1], z0[:, : t + 1], atol=1e-5)
        # the perturbation is visible from t+1 on, so the test is not vacuous
        assert np.max(np.abs(z1[:, t + 1:] - z0[:, t + 1:])) > 1e-6


def test_target_is_not_an_input(encoder):
    assert set(encoder.input.keys()) == set(F.ENCODER_INPUTS)
    x = _windows(2)
    x["target"] = x["signal"]
    with pytest.raises(F.TemporalContractError, match="target"):
        F.encode(encoder, x)


def test_save_load_roundtrip_without_decoder(encoder, tmp_path):
    import keras
    p = str(tmp_path / "enc.keras")
    encoder.save(p)
    back = keras.models.load_model(p)
    names = {l.name for l in back.layers}
    assert not any(n.startswith(("mtae_dec", "p2c_", "mask_token", "lineage")) for n in names)
    x = _windows(4)
    np.testing.assert_allclose(F.encode(back, x), F.encode(encoder, x), atol=1e-6)


def test_pooled_encoder_violates_contract():
    import keras
    from keras import layers
    ins = {k: keras.Input((T, C if k == "calendar" else 1), name=k) for k in F.ENCODER_INPUTS}
    h = layers.Concatenate()([ins[k] for k in F.ENCODER_INPUTS])
    pooled = keras.Model(ins, layers.Dense(D)(layers.GlobalAveragePooling1D()(h)))
    with pytest.raises(F.TemporalContractError, match="pooled"):
        F.assert_temporal_contract(pooled)


def test_input_shape_and_finiteness_guards():
    x = _windows(2)
    x["calendar"] = x["calendar"][..., :3]
    with pytest.raises(F.TemporalContractError):
        F.check_inputs(x, T, C)
    x = _windows(2)
    x["signal"][0, 3, 0] = np.nan
    with pytest.raises(F.TemporalContractError, match="non-finite"):
        F.check_inputs(x, T, C)


def test_mtae_masked_values_never_reach_encoder(mtae):
    _, trainer, _ = mtae
    x = _windows(4)
    pm = np.zeros((4, T, 1), np.float32)
    pm[:, 10] = 1.0
    args = [x[k] for k in F.ENCODER_INPUTS]
    r0 = trainer.predict(args + [pm], verbose=0)
    args[0] = args[0].copy()
    args[0][:, 10] += 100.0  # change only the masked value
    r1 = trainer.predict(args + [pm], verbose=0)
    np.testing.assert_allclose(r1, r0, atol=1e-5)


@pytest.mark.parametrize("fit", ["mtae", "p2c"])
def test_early_stopping_records_best(fit, mtae, p2c):
    res = {"mtae": mtae, "p2c": p2c}[fit][2]
    assert res.best_epoch >= 0 and np.isfinite(res.best_val)
    assert res.best_val == min(h["val"] for h in res.history)
    assert res.updates > 0


def test_mtae_restored_weights_reproduce_best_val(mtae):
    _, trainer, res = mtae
    v = _windows(16, seed=1)
    pm = F.pretext_masks(np.random.default_rng(0 + 1), 16, T, 0.5)  # same fixed validation masks
    y = np.concatenate([v["signal"], pm * v["observed_mask"]], axis=-1)
    got = float(trainer.evaluate([v[k] for k in F.ENCODER_INPUTS] + [pm], y, batch_size=16, verbose=0))
    assert got == pytest.approx(res.best_val, rel=1e-4)


def test_p2c_past_never_overlaps_current():
    rng = np.random.default_rng(0)
    cur, past, lin = F.sample_p2c_pairs(rng, 1000, T, 4 * T, 3, 2000)
    assert np.all(past + T <= cur)
    assert np.all(cur - past <= 4 * T) and np.all(past >= 0)
    assert set(np.unique(lin)) == {0, 1, 2}
    with pytest.raises(ValueError):
        F.sample_p2c_pairs(rng, 1000, T, T - 1, 3, 10)


def test_compatible_with_lane_d_interface(mtae, p2c):
    """Lane D's TemporalBatch feeds these encoders unchanged (skipped until lane D lands)."""
    U = pytest.importorskip("app.univariate_temporal")
    assert tuple(U.INPUT_NAMES) == F.ENCODER_INPUTS
    rng = np.random.default_rng(3)
    n = T + 40
    ts = (1_700_000_000 // 3600) * 3600 + 3600 * np.arange(n, dtype=np.int64)
    x = np.cumsum(rng.normal(size=n)).astype(np.float32)
    obs = rng.random(n) > 0.1
    norm = U.Normalization.fit(x, obs, np.arange(n))
    b = U.make_windows(ts, x, obs, np.arange(T - 1, T - 1 + 8), T, norm)
    assert b.calendar.shape[2] == C
    for enc in (mtae[0], p2c[0]):
        z = F.encode(enc, b.as_inputs())
        assert z.shape == (8, T, D)
