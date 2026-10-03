"""Typed adapter: materialized train NPZ -> .keras encoder + JSON manifest.

Input NPZ contract (only file ever read):
  x        float32 (samples, steps, channels), all values finite
  row_ids  optional 1-D integer array, length == samples (default: 0..samples-1)

The adapter trains a small Conv1D autoencoder on exactly the rows in the NPZ.
No validation/test data is read, and no other file is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

ARCHITECTURE_ID = "conv1d_ae_v1"


class NpzContractError(ValueError):
    pass


@dataclass(frozen=True)
class Manifest:
    input_sha256: str
    train_row_ids_sha256: str
    architecture_id: str
    seed: int
    n_rows: int
    steps: int
    channels: int
    latent_dim: int
    epochs: int
    reconstruction_mae: float
    reconstruction_mse: float
    weight_sha256: str
    encoder_path: str


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_ids_sha256(row_ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row_ids, dtype="<i8").tobytes()).hexdigest()


def load_train_npz(path: str):
    """Load and validate. Returns (x float32, row_ids int64)."""
    with np.load(path, allow_pickle=False) as z:
        if "x" not in z.files:
            raise NpzContractError("NPZ must contain array 'x'")
        x = z["x"]
        ids = z["row_ids"] if "row_ids" in z.files else None
    if x.dtype != np.float32:
        raise NpzContractError(f"x must be float32, got {x.dtype}")
    if x.ndim != 3:
        raise NpzContractError(f"x must be (samples, steps, channels), got ndim={x.ndim}")
    if min(x.shape) < 1:
        raise NpzContractError("x has an empty axis")
    if not np.isfinite(x).all():
        raise NpzContractError("x contains non-finite values (NaN/Inf)")
    if ids is None:
        ids = np.arange(x.shape[0], dtype=np.int64)
    else:
        if ids.ndim != 1 or ids.shape[0] != x.shape[0]:
            raise NpzContractError("row_ids must be 1-D with length == samples")
        if not np.issubdtype(ids.dtype, np.integer):
            raise NpzContractError("row_ids must be integer")
        ids = ids.astype(np.int64)
        if len(np.unique(ids)) != len(ids):
            raise NpzContractError("row_ids must be unique")
    return x, ids


def _build(steps: int, channels: int, latent_dim: int, filters: int):
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input(shape=(steps, channels), name="x")
    h = L.Conv1D(filters, 3, padding="same", activation="relu")(inp)
    h = L.Conv1D(filters, 3, padding="same", activation="relu")(h)
    h = L.Flatten()(h)
    z = L.Dense(latent_dim, name="latent")(h)
    encoder = tf.keras.Model(inp, z, name="encoder")
    zin = L.Input(shape=(latent_dim,))
    d = L.Dense(steps * filters, activation="relu")(zin)
    d = L.Reshape((steps, filters))(d)
    d = L.Conv1D(filters, 3, padding="same", activation="relu")(d)
    out = L.Conv1D(channels, 3, padding="same")(d)
    decoder = tf.keras.Model(zin, out, name="decoder")
    ae = tf.keras.Model(inp, decoder(encoder(inp)), name="autoencoder")
    return encoder, ae


def train_encoder(npz_path: str, out_dir: str, seed: int = 0, latent_dim: int = 16,
                  filters: int = 32, epochs: int = 20, batch_size: int = 128,
                  learning_rate: float = 1e-3) -> Manifest:
    x, ids = load_train_npz(npz_path)  # validates before any TF work
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    encoder, ae = _build(x.shape[1], x.shape[2], latent_dim, filters)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    ae.fit(x, x, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
    rec = ae.predict(x, batch_size=batch_size, verbose=0).astype(np.float64)
    err = rec - x.astype(np.float64)
    mae, mse = float(np.mean(np.abs(err))), float(np.mean(err ** 2))
    if not (np.isfinite(mae) and np.isfinite(mse)):
        raise NpzContractError("training diverged: non-finite reconstruction error")
    enc_path = os.path.join(out_dir, "encoder.keras")
    encoder.save(enc_path)
    wh = hashlib.sha256()
    for w in encoder.get_weights():
        wh.update(np.ascontiguousarray(w, dtype="<f4").tobytes())
    m = Manifest(
        input_sha256=sha256_file(npz_path), train_row_ids_sha256=row_ids_sha256(ids),
        architecture_id=ARCHITECTURE_ID, seed=seed, n_rows=int(x.shape[0]),
        steps=int(x.shape[1]), channels=int(x.shape[2]), latent_dim=latent_dim,
        epochs=epochs, reconstruction_mae=mae, reconstruction_mse=mse,
        weight_sha256=wh.hexdigest(), encoder_path=os.path.basename(enc_path))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(asdict(m), f, indent=2, sort_keys=True)
    return m


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--filters", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    a = p.parse_args(argv)
    m = train_encoder(a.train_npz, a.out_dir, a.seed, a.latent_dim, a.filters, a.epochs, a.batch_size)
    print(json.dumps(asdict(m), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
