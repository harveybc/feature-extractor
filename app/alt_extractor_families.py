"""Alternative univariate temporal extractor families (lane F), clean-room.

Two self-supervised families that share the univariate temporal interface

    signal        (B, T, 1)
    observed_mask (B, T, 1)
    delta_time    (B, T, 1)
    calendar      (B, T, C_known)
    -> latent     (B, T, D)

* MTAE - causal masked temporal autoencoder.  Point-level masking in the
  spirit of Ti-MAE (arXiv 2301.08871), but the encoder is causal: the latent
  at step t depends only on inputs at steps <= t.  Masked steps reach the
  encoder as a learned mask value with observed_mask = 0, never as their
  true value.  The loss is MSE on masked, actually-observed steps.
* P2C - past-to-current Siamese objective in the spirit of TimeSiam (ICML
  2024).  One shared causal encoder embeds a past window and a masked current
  window; a training-only decoder cross-attends from the current latent to the
  past latent plus a lineage embedding (distance bucket) and reconstructs the
  masked current steps.  The past window always ends at or before the start
  of the current window, so no current value can leak through it.

Neither family copies third-party code; both follow the papers' descriptions
only (see the lane F extractor family dossier for licences).  Neither is a
reproduction of the paper it cites and must not be labelled as one.

Contract (FS17): the encoder keeps T (no pooling), is exportable without its
decoder, and has no target input.  The future target is never an encoder
input; it may only supervise external probes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras  # noqa: E402
from keras import layers, ops  # noqa: E402

ENCODER_INPUTS = ("signal", "observed_mask", "delta_time", "calendar")
_PKG = "alt_extractor_families"


class TemporalContractError(ValueError):
    """Raised when inputs or a model violate the univariate temporal contract."""


# --------------------------------------------------------------------------
# Encoder (shared by both families)
# --------------------------------------------------------------------------
@dataclass
class EncoderConfig:
    window: int = 168
    calendar_dims: int = 6
    latent_dim: int = 8
    d_model: int = 32
    n_blocks: int = 2
    n_heads: int = 4
    kernel_size: int = 3
    dropout: float = 0.0


def build_causal_encoder(cfg: EncoderConfig, name: str = "encoder") -> keras.Model:
    """Causal Conv1D stem + causal self-attention blocks -> latent (B, T, D).

    Every layer is either per-step (Dense, LayerNorm), causal-padded Conv1D,
    or attention with a causal mask, so latent[:, t] depends on steps <= t.
    """
    T, C = cfg.window, cfg.calendar_dims
    ins = {
        "signal": keras.Input((T, 1), name="signal"),
        "observed_mask": keras.Input((T, 1), name="observed_mask"),
        "delta_time": keras.Input((T, 1), name="delta_time"),
        "calendar": keras.Input((T, C), name="calendar"),
    }
    x = layers.Concatenate(axis=-1, name="enc_concat")([ins[k] for k in ENCODER_INPUTS])
    x = layers.Conv1D(cfg.d_model, cfg.kernel_size, padding="causal", activation="gelu", name="enc_stem")(x)
    for i in range(cfg.n_blocks):
        a = layers.MultiHeadAttention(cfg.n_heads, max(1, cfg.d_model // cfg.n_heads),
                                      dropout=cfg.dropout, name=f"enc_mha_{i}")(x, x, use_causal_mask=True)
        x = layers.LayerNormalization(name=f"enc_ln_a_{i}")(layers.Add()([x, a]))
        f = layers.Dense(2 * cfg.d_model, activation="gelu", name=f"enc_ff1_{i}")(x)
        f = layers.Dense(cfg.d_model, name=f"enc_ff2_{i}")(f)
        x = layers.LayerNormalization(name=f"enc_ln_f_{i}")(layers.Add()([x, f]))
    latent = layers.Dense(cfg.latent_dim, name="latent")(x)
    return keras.Model(ins, latent, name=name)


def check_inputs(inputs: Mapping[str, np.ndarray], window: int, calendar_dims: int) -> Dict[str, np.ndarray]:
    """Validate an encoder input dict; reject any extra key (e.g. a target)."""
    extra = set(inputs) - set(ENCODER_INPUTS)
    if extra:
        raise TemporalContractError(f"encoder accepts only {ENCODER_INPUTS}; refused extra inputs {sorted(extra)}")
    missing = set(ENCODER_INPUTS) - set(inputs)
    if missing:
        raise TemporalContractError(f"missing encoder inputs {sorted(missing)}")
    out = {}
    for k in ENCODER_INPUTS:
        a = np.asarray(inputs[k], dtype=np.float32)
        want = calendar_dims if k == "calendar" else 1
        if a.ndim != 3 or a.shape[1] != window or a.shape[2] != want:
            raise TemporalContractError(f"{k}: expected (B,{window},{want}), got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise TemporalContractError(f"{k}: non-finite values")
        out[k] = a
    return out


def encode(encoder: keras.Model, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    T = encoder.input["signal"].shape[1]
    C = encoder.input["calendar"].shape[2]
    x = check_inputs(inputs, T, C)
    return np.asarray(encoder.predict(x, verbose=0))


def assert_temporal_contract(encoder: keras.Model) -> None:
    """FS17: inputs are exactly the four contract tensors; output keeps T."""
    names = set(encoder.input.keys()) if isinstance(encoder.input, dict) else set()
    if names != set(ENCODER_INPUTS):
        raise TemporalContractError(f"encoder inputs {sorted(names)} != {sorted(ENCODER_INPUTS)}")
    T = encoder.input["signal"].shape[1]
    out = encoder.output
    if len(out.shape) != 3 or out.shape[1] != T:
        raise TemporalContractError(f"pooled or non-temporal output {out.shape}; contract needs (B,{T},D)")


# --------------------------------------------------------------------------
# Training-only layers
# --------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package=_PKG)
class MaskToken(layers.Layer):
    """Replace masked steps' signal by a learned scalar and zero their observed flag."""

    def build(self, input_shape):
        self.token = self.add_weight(shape=(1,), initializer="zeros", name="mask_value")

    def call(self, signal, observed, pretext_mask):
        keep = 1.0 - pretext_mask
        return signal * keep + pretext_mask * self.token, observed * keep


@keras.saving.register_keras_serializable(package=_PKG)
class LineageAdd(layers.Layer):
    """Add a learned lineage (distance-bucket) embedding to every step of a latent."""

    def __init__(self, n_lineage: int, dim: int, **kw):
        super().__init__(**kw)
        self.n_lineage, self.dim = int(n_lineage), int(dim)
        self.emb = layers.Embedding(self.n_lineage, self.dim)

    def call(self, latent, lineage):
        e = self.emb(ops.cast(ops.reshape(lineage, (-1,)), "int32"))
        return latent + ops.expand_dims(e, 1)

    def get_config(self):
        return {**super().get_config(), "n_lineage": self.n_lineage, "dim": self.dim}


def masked_mse(y_true, y_pred):
    """y_true packs (target, weight) on the last axis; weight = pretext_mask * observed."""
    target, w = y_true[..., :1], y_true[..., 1:2]
    return ops.sum(w * ops.square(target - y_pred)) / (ops.sum(w) + 1e-8)


def _causal_head(z, d_model, name):
    h = layers.Conv1D(d_model, 3, padding="causal", activation="gelu", name=f"{name}_conv")(z)
    return layers.Dense(1, name=f"{name}_out")(h)


def build_mtae(cfg: EncoderConfig):
    enc = build_causal_encoder(cfg)
    T, C = cfg.window, cfg.calendar_dims
    sig = keras.Input((T, 1), name="signal")
    obs = keras.Input((T, 1), name="observed_mask")
    dt = keras.Input((T, 1), name="delta_time")
    cal = keras.Input((T, C), name="calendar")
    pm = keras.Input((T, 1), name="pretext_mask")
    s_in, o_in = MaskToken(name="mask_token")(sig, obs, pm)
    z = enc({"signal": s_in, "observed_mask": o_in, "delta_time": dt, "calendar": cal})
    rec = _causal_head(z, cfg.d_model, "mtae_dec")
    trainer = keras.Model([sig, obs, dt, cal, pm], rec, name="mtae_trainer")
    return enc, trainer


def build_p2c(cfg: EncoderConfig, n_lineage: int = 3):
    enc = build_causal_encoder(cfg)
    T, C = cfg.window, cfg.calendar_dims

    def window_inputs(prefix):
        return [keras.Input((T, 1), name=f"{prefix}_signal"), keras.Input((T, 1), name=f"{prefix}_observed_mask"),
                keras.Input((T, 1), name=f"{prefix}_delta_time"), keras.Input((T, C), name=f"{prefix}_calendar")]

    cur, past = window_inputs("cur"), window_inputs("past")
    pm = keras.Input((T, 1), name="pretext_mask")
    lin = keras.Input((1,), dtype="int32", name="lineage")
    s_in, o_in = MaskToken(name="mask_token")(cur[0], cur[1], pm)
    z_cur = enc({"signal": s_in, "observed_mask": o_in, "delta_time": cur[2], "calendar": cur[3]})
    z_past = enc(dict(zip(ENCODER_INPUTS, past)))
    z_past = LineageAdd(n_lineage, cfg.latent_dim, name="lineage_add")(z_past, lin)
    q = layers.Dense(cfg.d_model, name="p2c_q")(z_cur)
    kv = layers.Dense(cfg.d_model, name="p2c_kv")(z_past)
    # Causal self-attention over the current latent, then cross-attention to the
    # (strictly earlier) past window; the past is wholly before the current
    # window, so attending to all of it does not see any current value.
    sa = layers.MultiHeadAttention(cfg.n_heads, max(1, cfg.d_model // cfg.n_heads), name="p2c_self")(q, q, use_causal_mask=True)
    h = layers.LayerNormalization(name="p2c_ln1")(layers.Add()([q, sa]))
    ca = layers.MultiHeadAttention(cfg.n_heads, max(1, cfg.d_model // cfg.n_heads), name="p2c_cross")(h, kv)
    h = layers.LayerNormalization(name="p2c_ln2")(layers.Add()([h, ca]))
    rec = _causal_head(h, cfg.d_model, "p2c_dec")
    trainer = keras.Model(cur + past + [pm, lin], rec, name="p2c_trainer")
    return enc, trainer


# --------------------------------------------------------------------------
# Sampling (TRAIN-only arrays are the caller's responsibility)
# --------------------------------------------------------------------------
def pretext_masks(rng: np.random.Generator, n: int, window: int, ratio: float) -> np.ndarray:
    return (rng.random((n, window, 1)) < ratio).astype(np.float32)


def windows_from_series(series: Mapping[str, np.ndarray], window: int, starts: np.ndarray) -> Dict[str, np.ndarray]:
    """Slice contiguous (L, ch) arrays for each contract input into (N, T, ch) windows."""
    idx = np.asarray(starts)[:, None] + np.arange(window)[None, :]
    return {k: np.asarray(series[k], dtype=np.float32)[idx] for k in ENCODER_INPUTS}


def sample_p2c_pairs(rng: np.random.Generator, length: int, window: int, max_lag: int, n_lineage: int, n: int):
    """Return (cur_starts, past_starts, lineage).

    Lag = cur_start - past_start in [window, max_lag], so the past window ends
    at or before the current window begins.  Lineage buckets the lag evenly.
    """
    if max_lag < window:
        raise ValueError("max_lag must be >= window so the past never overlaps the current window")
    if length < max_lag + window:
        raise ValueError("series too short for the requested max_lag")
    cur = rng.integers(max_lag, length - window + 1, size=n)
    lag = rng.integers(window, max_lag + 1, size=n)
    span = max_lag - window + 1
    lineage = np.minimum((lag - window) * n_lineage // span, n_lineage - 1).astype(np.int32)
    return cur, cur - lag, lineage


# --------------------------------------------------------------------------
# Training with early stopping and best-checkpoint restore
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    history: list = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")
    updates: int = 0


def _fit(trainer, make_batch, epochs, patience, batch_size, lr, seed) -> FitResult:
    trainer.compile(optimizer=keras.optimizers.Adam(lr), loss=masked_mse)
    rng = np.random.default_rng(seed)
    val_x, val_y = make_batch("val", np.random.default_rng(seed + 1))  # fixed validation masks
    res, best_w, wait = FitResult(), None, 0
    for ep in range(epochs):
        x, y = make_batch("train", rng)  # fresh masks every epoch
        h = trainer.fit(x, y, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        res.updates += int(np.ceil(len(y) / batch_size))
        v = float(trainer.evaluate(val_x, val_y, batch_size=batch_size, verbose=0))
        res.history.append({"epoch": ep, "train": float(h.history["loss"][-1]), "val": v})
        if v < res.best_val:
            res.best_val, res.best_epoch, best_w, wait = v, ep, trainer.get_weights(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_w is not None:
        trainer.set_weights(best_w)
    return res


def fit_mtae(cfg: EncoderConfig, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray], *,
             mask_ratio=0.5, epochs=50, patience=5, batch_size=64, lr=1e-3, seed=0):
    """train/val: window dicts (N,T,ch) of the four contract inputs (TRAIN-derived only)."""
    keras.utils.set_random_seed(seed)
    tr = check_inputs(train, cfg.window, cfg.calendar_dims)
    va = check_inputs(val, cfg.window, cfg.calendar_dims)
    enc, trainer = build_mtae(cfg)

    def make_batch(split, rng):
        d = tr if split == "train" else va
        pm = pretext_masks(rng, len(d["signal"]), cfg.window, mask_ratio)
        x = [d["signal"], d["observed_mask"], d["delta_time"], d["calendar"], pm]
        return x, np.concatenate([d["signal"], pm * d["observed_mask"]], axis=-1)

    return enc, trainer, _fit(trainer, make_batch, epochs, patience, batch_size, lr, seed)


def fit_p2c(cfg: EncoderConfig, train_series: Mapping[str, np.ndarray], val_series: Mapping[str, np.ndarray], *,
            max_lag: Optional[int] = None, n_lineage=3, pairs_per_epoch=512, mask_ratio=0.5,
            epochs=50, patience=5, batch_size=64, lr=1e-3, seed=0):
    """train_series/val_series: contiguous (L, ch) arrays per contract input."""
    keras.utils.set_random_seed(seed)
    T = cfg.window
    max_lag = max_lag or 3 * T
    enc, trainer = build_p2c(cfg, n_lineage)

    def make_batch(split, rng):
        s = train_series if split == "train" else val_series
        L = len(s["signal"])
        cur, past, lin = sample_p2c_pairs(rng, L, T, max_lag, n_lineage, pairs_per_epoch)
        c = check_inputs(windows_from_series(s, T, cur), T, cfg.calendar_dims)
        p = check_inputs(windows_from_series(s, T, past), T, cfg.calendar_dims)
        pm = pretext_masks(rng, len(cur), T, mask_ratio)
        x = [c[k] for k in ENCODER_INPUTS] + [p[k] for k in ENCODER_INPUTS] + [pm, lin[:, None]]
        return x, np.concatenate([c["signal"], pm * c["observed_mask"]], axis=-1)

    return enc, trainer, _fit(trainer, make_batch, epochs, patience, batch_size, lr, seed)
