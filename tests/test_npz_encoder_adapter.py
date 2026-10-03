import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import pytest

from app import npz_encoder_adapter as A


def _save(tmp_path, x, name="train.npz", **kw):
    p = str(tmp_path / name)
    np.savez(p, x=x, **kw)
    return p


def _x(n=24, s=8, c=3):
    return np.random.default_rng(0).normal(size=(n, s, c)).astype(np.float32)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite(tmp_path, bad):
    x = _x()
    x[3, 2, 1] = bad
    with pytest.raises(A.NpzContractError, match="non-finite"):
        A.train_encoder(_save(tmp_path, x), str(tmp_path / "o"), epochs=1)
    assert not (tmp_path / "o").exists()


def test_rejects_wrong_dtype_and_rank(tmp_path):
    with pytest.raises(A.NpzContractError):
        A.load_train_npz(_save(tmp_path, _x().astype(np.float64)))
    with pytest.raises(A.NpzContractError):
        A.load_train_npz(_save(tmp_path, _x()[0], name="b.npz"))


def test_manifest_and_only_supplied_rows(tmp_path, monkeypatch):
    x = _x()
    ids = np.arange(100, 100 + len(x))
    p = _save(tmp_path, x, row_ids=ids)
    opened = []
    real_load = np.load
    monkeypatch.setattr(np, "load", lambda f, *a, **k: (opened.append(str(f)), real_load(f, *a, **k))[1])
    m = A.train_encoder(p, str(tmp_path / "o"), seed=3, epochs=1, latent_dim=4, filters=8)
    assert opened == [p]  # no other file read
    assert m.n_rows == len(x) and m.train_row_ids_sha256 == A.row_ids_sha256(ids)
    assert m.input_sha256 == A.sha256_file(p)
    assert m.seed == 3 and m.architecture_id == A.ARCHITECTURE_ID
    assert m.reconstruction_mae >= 0 and m.reconstruction_mse >= 0
    man = json.load(open(tmp_path / "o" / "manifest.json"))
    assert man["weight_sha256"] == m.weight_sha256 and len(m.weight_sha256) == 64
    assert (tmp_path / "o" / "encoder.keras").exists()
    import tensorflow as tf
    enc = tf.keras.models.load_model(str(tmp_path / "o" / "encoder.keras"))
    assert enc.predict(x, verbose=0).shape == (len(x), 4)
    assert A.row_ids_sha256(ids[:-1]) != m.train_row_ids_sha256


def test_materialized_windows_contract_and_split_guard(tmp_path):
    import numpy as np, pytest
    from app.npz_encoder_adapter import load_train_npz, NpzContractError
    x = np.random.RandomState(0).rand(6, 4, 3).astype("float32")
    ids = np.array([f"r{i}" for i in range(6)])
    ok = tmp_path / "ok.npz"
    np.savez(ok, windows=x, row_ids=ids, split=np.array("train"))
    lx, lid = load_train_npz(str(ok))
    assert lx.shape == (6, 4, 3) and list(lid) == list(ids)
    bad = tmp_path / "bad.npz"
    np.savez(bad, windows=x, row_ids=ids, split=np.array("validation"))
    with pytest.raises(NpzContractError):
        load_train_npz(str(bad))
