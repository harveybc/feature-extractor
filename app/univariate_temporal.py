"""Univariate temporal extractor: one causal series in, a temporal latent out.

Operational interface (lane D, canonical selection-first plan 2026-10-03):

    signal        (B, T, 1)        standardized with TRAIN-fit statistics, 0 where unobserved
    observed_mask (B, T, 1)        1 observed / 0 missing (no fabricated observations)
    delta_time    (B, T, 1)        log1p(hours since the previous observation), causal
    calendar      (B, T, C_known)  sin/cos hour, day-of-week, day-of-year (+ columns published at t)
    latent        (B, T, D)        time axis preserved: no pooling, flatten or striding

The target is never an input of the operational encoder (FS03). It may only
supervise probes (see temporal_extractor_metrics) or an offline generator.

Families share this interface:
    identity  raw control: [signal, observed_mask, delta_time], D=3 declared, no weights
    random    untrained encoder of the identical architecture (seeded)
    ae        causal Conv1D stem + residual dilated TCN + per-instant calendar fusion,
              trained with a causal decoder on masked reconstruction
    dae       same, with a declared corruption (gaussian on observed values) at training only
    masked_temporal_ae       lane F clean-room causal masked temporal AE (app.alt_extractor_families)
    past_to_current_siamese  lane F clean-room past-to-current siamese (app.alt_extractor_families)
                             needs contiguous TRAIN windows and >= max_lag of history before each one

Interface status: FINAL for lanes E/F as of 2026-10-03 (ut_donor.v1, ps2_batch.v1).

Plain AE caveat recorded in every manifest: with D >= input channels per instant
there is no per-instant bottleneck, so reconstruction can be identity-like.
Reconstruction is a diagnostic, never a selection criterion (FS06).

Donors are exported encoder-only (`encoder.keras` + `donor_manifest.json`) and are
loadable as R0 (same architecture, fresh random weights), R1 (donor weights, frozen)
or R2 (donor weights, trainable). Identity is verified before any weight is used.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from app.npz_encoder_adapter import NpzContractError, row_ids_sha256, sha256_file

DONOR_SCHEMA = "ut_donor.v1"
ARCHITECTURE_ID = "ut_causal_tcn_v1"
IDENTITY_ARCHITECTURE_ID = "ut_identity_v1"
INPUT_NAMES = ("signal", "observed_mask", "delta_time", "calendar")
REGIMES = ("R0", "R1", "R2")
CALENDAR_SPEC = ("sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_doy", "cos_doy")
_TARGET_NAME = re.compile(r"^(y|y_.*|target.*|label.*|future.*)$", re.IGNORECASE)

__all__ = ["row_ids_sha256", "sha256_file"]  # re-exported from the single adapter line


class ContractError(NpzContractError):
    pass


class TargetLeakError(ContractError):
    pass


class CalendarLeakError(ContractError):
    pass


class TemporalContractError(ContractError):
    pass


class FoldScopeError(ContractError):
    pass


class DonorIdentityError(ContractError):
    pass


class SlotNotImplemented(NotImplementedError):
    pass


class TrainingDiverged(ContractError):
    pass


# --------------------------------------------------------------------------- guards
def check_no_target(mapping) -> None:
    """Refuse any key that names a target/label/future quantity."""
    for k in mapping:
        if _TARGET_NAME.match(str(k)):
            raise TargetLeakError(f"{k!r} is a target-like name; targets never enter the operational encoder")


def assert_temporal_output(z, batch: int, window: int) -> None:
    shape = tuple(np.shape(z))
    if len(shape) != 3 or shape[0] != batch or shape[1] != window:
        raise TemporalContractError(f"latent must be (B={batch}, T={window}, D); got {shape}")


def assert_temporal_encoder(model, window: int) -> None:
    shape = tuple(model.output.shape)
    if len(shape) != 3 or shape[1] != window:
        raise TemporalContractError(f"encoder output must be (B, {window}, D); got {shape} (pooled/flattened)")


# --------------------------------------------------------------------------- calendar and time
def calendar_features(ts: np.ndarray) -> np.ndarray:
    """(N,) UTC epoch seconds -> (N, 6) float32. Row t depends on timestamp t only."""
    ts = np.asarray(ts, dtype=np.int64)
    hour = (ts // 3600) % 24
    days = ts // 86400
    dow = (days + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    dt = ts.astype("datetime64[s]")
    year = dt.astype("datetime64[Y]")
    doy = (dt.astype("datetime64[D]") - year.astype("datetime64[D]")).astype(np.int64)
    ndays = ((year + 1).astype("datetime64[D]") - year.astype("datetime64[D]")).astype(np.int64)
    a_h, a_w, a_y = 2 * np.pi * hour / 24.0, 2 * np.pi * dow / 7.0, 2 * np.pi * doy / ndays
    out = np.stack([np.sin(a_h), np.cos(a_h), np.sin(a_w), np.cos(a_w), np.sin(a_y), np.cos(a_y)], axis=1)
    return out.astype(np.float32)


def known_calendar_columns(ts: np.ndarray, columns: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Session/holiday style columns, accepted only if published at or before t."""
    ts = np.asarray(ts, dtype=np.int64)
    out = []
    for name in sorted(columns):
        values, published_at = columns[name]
        values = np.asarray(values, dtype=np.float32)
        published_at = np.asarray(published_at, dtype=np.int64)
        if values.shape != ts.shape or published_at.shape != ts.shape:
            raise ContractError(f"calendar column {name!r} must align with timestamps")
        if not np.isfinite(values).all():
            raise ContractError(f"calendar column {name!r} has non-finite values")
        late = np.nonzero(published_at > ts)[0]
        if late.size:
            raise CalendarLeakError(f"calendar column {name!r} published after t at {late.size} rows (first idx {late[0]})")
        out.append(values)
    if not out:
        return np.zeros((ts.shape[0], 0), np.float32)
    return np.stack(out, axis=1)


def delta_time_series(ts: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """log1p(hours since the last observation strictly before t). Causal by construction."""
    ts = np.asarray(ts, dtype=np.int64)
    obs = np.asarray(observed, dtype=bool)
    gaps = np.zeros(ts.shape[0], np.float64)
    gaps[1:] = np.diff(ts) / 3600.0
    d = np.zeros(ts.shape[0], np.float64)
    for t in range(1, ts.shape[0]):
        d[t] = gaps[t] + (0.0 if obs[t - 1] else d[t - 1])
    return np.log1p(d).astype(np.float32)


# --------------------------------------------------------------------------- folds and normalization
@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    split: str
    fit: Tuple[int, int]
    val: Tuple[int, int]


def fold_anchor_indices(ts: np.ndarray, rng: Sequence[int]) -> np.ndarray:
    a, b = int(rng[0]), int(rng[1])
    return np.nonzero((ts >= a) & (ts <= b))[0]


def covered_indices(anchor_idx: np.ndarray, window: int) -> np.ndarray:
    if anchor_idx.size == 0:
        return anchor_idx
    m = np.zeros(int(anchor_idx.max()) + 1, bool)
    for i in anchor_idx:
        m[max(0, i - window + 1): i + 1] = True
    return np.nonzero(m)[0]


def validate_fold(fold: FoldSpec, ts: np.ndarray, train_end_ts: int, window: int) -> None:
    if fold.split != "train":
        raise FoldScopeError(f"fold {fold.fold_id}: split must be 'train', got {fold.split!r}")
    for name, (a, b) in (("fit", fold.fit), ("val", fold.val)):
        if a > b:
            raise FoldScopeError(f"fold {fold.fold_id}: empty {name} range")
        if b > train_end_ts:
            raise FoldScopeError(f"fold {fold.fold_id}: {name} reaches beyond TRAIN end")
    fi, vi = fold_anchor_indices(ts, fold.fit), fold_anchor_indices(ts, fold.val)
    if fi.size == 0 or vi.size == 0:
        raise FoldScopeError(f"fold {fold.fold_id}: no anchors in fit or val")
    if fi.min() < window - 1 or vi.min() < window - 1:
        raise FoldScopeError(f"fold {fold.fold_id}: anchors need {window - 1} prior rows of history")
    after = vi.min() - (window - 1) > fi.max()
    before = fi.min() - (window - 1) > vi.max()
    if not (after or before):
        raise FoldScopeError(f"fold {fold.fold_id}: validation windows overlap fit rows (purge >= window)")


@dataclass(frozen=True)
class Normalization:
    mean: float
    std: float
    n: int
    constant: bool

    @classmethod
    def fit(cls, x: np.ndarray, observed: np.ndarray, idx: np.ndarray) -> "Normalization":
        idx = np.asarray(idx)
        v = np.asarray(x, np.float64)[idx][np.asarray(observed, bool)[idx]]
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ContractError("no observed TRAIN values to fit normalization")
        mu, sd = float(v.mean()), float(v.std())
        const = not sd > 1e-12
        return cls(mu, 1.0 if const else sd, int(v.size), const)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, np.float64) - self.mean) / self.std).astype(np.float32)


# --------------------------------------------------------------------------- batch
@dataclass
class TemporalBatch:
    signal: np.ndarray
    observed_mask: np.ndarray
    delta_time: np.ndarray
    calendar: np.ndarray
    row_ids: np.ndarray
    anchor_ts: np.ndarray

    def __post_init__(self):
        self.validate()

    def __len__(self):
        return int(self.signal.shape[0])

    def validate(self) -> None:
        s, m, d, c = self.signal, self.observed_mask, self.delta_time, self.calendar
        for name, a in zip(INPUT_NAMES, (s, m, d, c)):
            if not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.ndim != 3:
                raise ContractError(f"{name} must be float32 (B, T, C)")
            if not np.isfinite(a).all():
                raise ContractError(f"{name} has non-finite values")
            if a.shape[:2] != s.shape[:2]:
                raise TemporalContractError(f"{name} grid {a.shape[:2]} != signal grid {s.shape[:2]}")
        if s.shape[2] != 1 or m.shape[2] != 1 or d.shape[2] != 1:
            raise ContractError("signal, observed_mask and delta_time must have one channel")
        if not np.isin(m, (0.0, 1.0)).all():
            raise ContractError("observed_mask must be 0/1")
        if np.any(s[m == 0] != 0):
            raise ContractError("signal must be 0 where unobserved (no fabricated observations)")
        if np.any(d < 0):
            raise ContractError("delta_time must be >= 0")
        if len(self.row_ids) != s.shape[0] or len(self.anchor_ts) != s.shape[0]:
            raise ContractError("row_ids/anchor_ts length must equal batch size")

    def as_inputs(self) -> Dict[str, np.ndarray]:
        return {"signal": self.signal, "observed_mask": self.observed_mask,
                "delta_time": self.delta_time, "calendar": self.calendar}

    def copy_arrays(self) -> "TemporalBatch":
        return TemporalBatch(*(np.array(getattr(self, f)) for f in
                               ("signal", "observed_mask", "delta_time", "calendar", "row_ids", "anchor_ts")))

    def subset(self, idx) -> "TemporalBatch":
        return TemporalBatch(*(getattr(self, f)[idx] for f in
                               ("signal", "observed_mask", "delta_time", "calendar", "row_ids", "anchor_ts")))

    @classmethod
    def from_inputs(cls, d: Dict[str, np.ndarray], row_ids=None, anchor_ts=None) -> "TemporalBatch":
        check_no_target(d)
        extra = set(d) - set(INPUT_NAMES)
        if extra or set(INPUT_NAMES) - set(d):
            raise ContractError(f"inputs must be exactly {INPUT_NAMES}; extra={sorted(extra)}")
        n = np.shape(d["signal"])[0]
        ids = np.arange(n, dtype=np.int64) if row_ids is None else np.asarray(row_ids)
        ats = ids if anchor_ts is None else np.asarray(anchor_ts)
        return cls(*(np.asarray(d[k]) for k in INPUT_NAMES), ids, ats)


def make_windows(ts, x, observed, anchors, window, norm: Normalization, calendar=None) -> TemporalBatch:
    """Right-edge windows ending at each anchor index. Uses rows <= anchor only."""
    ts = np.asarray(ts, np.int64)
    obs = np.asarray(observed, bool) & np.isfinite(np.asarray(x, np.float64))
    anchors = np.asarray(anchors, np.int64)
    if anchors.size and anchors.min() < window - 1:
        raise TemporalContractError("anchor without a full window of history")
    cal = calendar_features(ts) if calendar is None else np.asarray(calendar, np.float32)
    xn = np.where(obs, norm.apply(np.where(obs, x, 0.0)), 0.0).astype(np.float32)
    dt = delta_time_series(ts, obs)
    idx = anchors[:, None] - (window - 1) + np.arange(window)[None, :]
    return TemporalBatch(xn[idx][..., None], obs.astype(np.float32)[idx][..., None], dt[idx][..., None],
                         cal[idx], ts[anchors].copy(), ts[anchors].copy())


# --------------------------------------------------------------------------- architecture
@dataclass(frozen=True)
class ArchConfig:
    window: int = 168
    calendar_dim: int = 6
    latent_dim: int = 8
    filters: int = 16
    kernel_size: int = 3
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    decoder_filters: int = 16
    decoder_dilations: Tuple[int, ...] = (1, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dilations"], d["decoder_dilations"] = list(self.dilations), list(self.decoder_dilations)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ArchConfig":
        d = dict(d)
        d["dilations"], d["decoder_dilations"] = tuple(d["dilations"]), tuple(d["decoder_dilations"])
        return cls(**d)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * (1 + 2 * sum(self.dilations))


def _inputs(cfg: ArchConfig):
    import keras
    return {"signal": keras.Input((cfg.window, 1), name="signal"),
            "observed_mask": keras.Input((cfg.window, 1), name="observed_mask"),
            "delta_time": keras.Input((cfg.window, 1), name="delta_time"),
            "calendar": keras.Input((cfg.window, cfg.calendar_dim), name="calendar")}


def build_encoder(cfg: ArchConfig, name: str = "ut_encoder"):
    import keras
    L = keras.layers
    inp = _inputs(cfg)
    F, k = cfg.filters, cfg.kernel_size
    x = L.Concatenate(name="series_channels")([inp["signal"], inp["observed_mask"], inp["delta_time"]])
    h = L.Conv1D(F, k, padding="causal", name="stem")(x)
    c = L.Conv1D(F, 1, name="calendar_proj")(inp["calendar"])  # per-instant projection
    h = L.Conv1D(F, 1, activation="relu", name="fuse")(L.Concatenate(name="fuse_concat")([h, c]))
    for i, d in enumerate(cfg.dilations):
        u = L.Conv1D(F, k, padding="causal", dilation_rate=d, activation="relu", name=f"tcn{i}_a")(h)
        u = L.Conv1D(F, k, padding="causal", dilation_rate=d, name=f"tcn{i}_b")(u)
        h = L.Activation("relu", name=f"tcn{i}_out")(L.Add(name=f"tcn{i}_add")([h, u]))
    z = L.Conv1D(cfg.latent_dim, 1, name="latent")(h)
    model = keras.Model(inp, z, name=name)
    assert_temporal_encoder(model, cfg.window)
    return model


def build_decoder(cfg: ArchConfig):
    import keras
    L = keras.layers
    z = keras.Input((cfg.window, cfg.latent_dim), name="latent_in")
    h = z
    for i, d in enumerate(cfg.decoder_dilations):
        h = L.Conv1D(cfg.decoder_filters, cfg.kernel_size, padding="causal", dilation_rate=d,
                     activation="relu", name=f"dec{i}")(h)
    out = L.Conv1D(1, 1, name="reconstruction")(h)
    return keras.Model(z, out, name="ut_decoder")


def _corruption_layer(sigma: float, seed: int):
    import keras

    class GaussianObservedCorruption(keras.layers.Layer):
        """Adds N(0, sigma) to observed values at training only; missing points stay 0."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.sigma = float(sigma)
            self.seed_generator = keras.random.SeedGenerator(seed)

        def call(self, inputs, training=None):
            s, m = inputs
            if not training:
                return s
            return s + keras.random.normal(keras.ops.shape(s), stddev=self.sigma, seed=self.seed_generator) * m

    return GaussianObservedCorruption(name="declared_corruption")


def weights_sha256(model) -> str:
    h = hashlib.sha256()
    for v in model.weights:
        if "seed_generator" in getattr(v, "path", v.name):
            continue
        a = np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype="<f4")
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def one_update_step(encoder, batch: TemporalBatch, lr: float = 1e-2) -> int:
    """One SGD step on a dummy latent objective; returns the number of trainable tensors."""
    import tensorflow as tf
    import keras
    inputs = {k: tf.convert_to_tensor(v) for k, v in batch.as_inputs().items()}
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(encoder(inputs, training=True) - 1.0))
    tw = encoder.trainable_weights
    if tw:
        keras.optimizers.SGD(lr).apply_gradients(zip(tape.gradient(loss, tw), tw))
    return len(tw)


# --------------------------------------------------------------------------- families
@dataclass(frozen=True)
class FamilySpec:
    name: str
    trainable: bool
    has_decoder: bool
    objective: str
    status: str = "IMPLEMENTED"
    default_corruption: Optional[dict] = None


IMPLEMENTED_LANE_F = "IMPLEMENTED_LANE_F"
FAMILIES: Dict[str, FamilySpec] = {
    "identity": FamilySpec("identity", False, False, "none (raw control)"),
    "random": FamilySpec("random", False, False, "none (untrained, identical architecture)"),
    "ae": FamilySpec("ae", True, True, "masked causal reconstruction"),
    "dae": FamilySpec("dae", True, True, "masked causal reconstruction under declared corruption",
                      default_corruption={"type": "gaussian_observed", "sigma": 0.1}),
    "masked_temporal_ae": FamilySpec("masked_temporal_ae", True, False,
                                     "masked causal reconstruction of pretext-masked observed steps (lane F)",
                                     status=IMPLEMENTED_LANE_F,
                                     default_corruption={"type": "pretext_mask", "ratio": 0.5}),
    "past_to_current_siamese": FamilySpec("past_to_current_siamese", True, False,
                                          "masked current reconstruction attending to a strictly earlier "
                                          "past window (lane F)", status=IMPLEMENTED_LANE_F,
                                          default_corruption={"type": "pretext_mask", "ratio": 0.5}),
}
RUNNABLE_STATUSES = ("IMPLEMENTED", IMPLEMENTED_LANE_F)


@dataclass(frozen=True)
class EarlyStopConfig:
    max_epochs: int = 200
    patience: int = 10
    min_delta: float = 0.0
    monitor: str = "masked_val_mse"


@dataclass(frozen=True)
class TrainScope:
    kind: str  # TRAIN_ONLY | FOREIGN_CORPUS
    split: str
    fold_id: str
    fit_first_ts: int
    fit_last_ts: int
    input_sha256: str
    train_row_ids_sha256: str

    def validate(self) -> None:
        if self.kind not in ("TRAIN_ONLY", "FOREIGN_CORPUS"):
            raise DonorIdentityError(f"unknown train scope kind {self.kind!r}")
        if self.kind == "TRAIN_ONLY" and self.split != "train":
            raise DonorIdentityError("TRAIN_ONLY scope requires split == 'train'")


def assert_train_only(manifest: dict, expected_input_sha256: str) -> None:
    sc = manifest.get("train_scope") or {}
    if sc.get("kind") != "TRAIN_ONLY" or sc.get("split") != "train":
        raise DonorIdentityError("donor is not TRAIN_ONLY")
    if manifest.get("input_sha256") != expected_input_sha256:
        raise DonorIdentityError("donor was fitted on a different corpus; it cannot be labeled TRAIN_ONLY here")


def _atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


class UnivariateTemporalExtractor:
    def __init__(self, family: str, cfg: ArchConfig, seed: int, learning_rate: float = 1e-3,
                 batch_size: int = 64, corruption: Optional[dict] = None):
        spec = FAMILIES[family]
        self.family, self.spec, self.cfg, self.seed = family, spec, cfg, int(seed)
        self.learning_rate, self.batch_size = float(learning_rate), int(batch_size)
        self.corruption = corruption if corruption is not None else spec.default_corruption
        self.fit_report: Optional[dict] = None
        self.encoder = self.decoder = self.training_model = None
        if family == "identity":
            self.architecture_id, self.latent_dim = IDENTITY_ARCHITECTURE_ID, 3
            return
        import keras
        keras.utils.set_random_seed(self.seed)
        self.architecture_id, self.latent_dim = ARCHITECTURE_ID, cfg.latent_dim
        self.encoder = build_encoder(cfg)
        if spec.has_decoder:
            self.decoder = build_decoder(cfg)
            inp = _inputs(cfg)
            enc_in = dict(inp)
            if self.corruption:
                if self.corruption.get("type") != "gaussian_observed":
                    raise ContractError(f"undeclared corruption {self.corruption!r}")
                enc_in["signal"] = _corruption_layer(self.corruption["sigma"], self.seed)(
                    [inp["signal"], inp["observed_mask"]])
            out = self.decoder(self.encoder(enc_in))
            self.training_model = keras.Model(inp, out, name=f"ut_{family}_training")
            self.training_model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")

    def extra_manifest(self) -> dict:
        return {}

    # -- encode
    def encode_inputs(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        return self.encode(TemporalBatch.from_inputs(d))

    def encode(self, batch: TemporalBatch) -> np.ndarray:
        if batch.signal.shape[1] != self.cfg.window or batch.calendar.shape[2] != self.cfg.calendar_dim:
            raise TemporalContractError(f"batch grid {batch.calendar.shape[1:]} != configured "
                                        f"({self.cfg.window}, {self.cfg.calendar_dim})")
        if self.family == "identity":
            z = np.concatenate([batch.signal, batch.observed_mask, batch.delta_time], axis=-1)
        else:
            z = np.asarray(self.encoder.predict(batch.as_inputs(), batch_size=256, verbose=0), np.float32)
        assert_temporal_output(z, len(batch), self.cfg.window)
        return z

    def reconstruct(self, batch: TemporalBatch) -> Optional[np.ndarray]:
        if self.training_model is None:
            return None
        return np.asarray(self.training_model.predict(batch.as_inputs(), batch_size=256, verbose=0), np.float32)

    def validation_loss(self, batch: TemporalBatch) -> float:
        r = self.reconstruct(batch)
        m = batch.observed_mask.astype(np.float64)
        return float(np.sum(m * (r.astype(np.float64) - batch.signal) ** 2) / max(np.sum(m), 1.0))

    # -- fit
    def fit(self, train: TemporalBatch, val: TemporalBatch, es: EarlyStopConfig) -> dict:
        if not self.spec.trainable:
            self.fit_report = {"stop_reason": "NOT_TRAINED", "epochs_run": 0, "updates": 0}
            return self.fit_report
        import keras
        ext = self

        class BestCheckpoint(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best, self.best_epoch, self.best_weights, self.wait = np.inf, -1, None, 0
                self.hist, self.digests, self.stop_reason, self.updates_at_best = [], [], "max_epochs", 0

            def on_epoch_end(self, epoch, logs=None):
                vl = ext.validation_loss(val)
                self.hist.append(vl)
                self.digests.append(weights_sha256(self.model))
                if np.isfinite(vl) and vl < self.best - es.min_delta:
                    self.best, self.best_epoch, self.wait = vl, epoch, 0
                    self.best_weights = [np.array(w) for w in self.model.get_weights()]
                    self.updates_at_best = int(self.model.optimizer.iterations.numpy())
                else:
                    self.wait += 1
                    if self.wait >= es.patience:
                        self.stop_reason, self.model.stop_training = "patience", True

            def on_train_end(self, logs=None):
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

        cb = BestCheckpoint()
        t0 = time.perf_counter()
        self.training_model.fit(train.as_inputs(), train.signal, sample_weight=train.observed_mask[..., 0],
                                batch_size=self.batch_size, epochs=es.max_epochs, shuffle=True,
                                callbacks=[cb], verbose=0)
        if cb.best_weights is None:
            raise TrainingDiverged("validation loss never finite; no checkpoint to restore")
        self.fit_report = {
            "stop_reason": cb.stop_reason, "epochs_run": len(cb.hist), "best_epoch": cb.best_epoch,
            "best_val_loss": float(cb.best), "val_loss_history": [float(v) for v in cb.hist],
            "weights_sha256_by_epoch": cb.digests,
            "updates": int(self.training_model.optimizer.iterations.numpy()),
            "updates_at_best": cb.updates_at_best, "fit_wall_seconds": time.perf_counter() - t0,
            "n_train_windows": len(train), "n_val_windows": len(val), "early_stop": asdict(es),
            "restored_best_checkpoint": True,
        }
        return self.fit_report

    # -- donor
    def export_donor(self, out_dir: str, scope: TrainScope, feature_id: str,
                     normalization: Optional[Normalization] = None) -> dict:
        scope.validate()
        if self.encoder is None:
            raise ContractError("identity control has no weights to donate")
        os.makedirs(out_dir, exist_ok=True)
        enc_path = os.path.join(out_dir, "encoder.keras")
        self.encoder.save(enc_path)
        rep = dict(self.fit_report or {})
        rep.pop("weights_sha256_by_epoch", None)
        rep.pop("history", None)
        man = {
            "schema": DONOR_SCHEMA, "family": self.family, "architecture_id": self.architecture_id,
            "arch_config": self.cfg.to_dict(), "arch_config_sha256": self.cfg.sha256(),
            "receptive_field_steps": self.cfg.receptive_field(),
            "seed": self.seed, "feature_id": feature_id, "window": self.cfg.window,
            "calendar_dim": self.cfg.calendar_dim, "latent_dim": self.cfg.latent_dim,
            "input_names": list(INPUT_NAMES), "calendar_spec": list(CALENDAR_SPEC),
            "objective": self.spec.objective, "trained": bool(self.spec.trainable and self.fit_report),
            "corruption": self.corruption, "fit_report": rep,
            "per_instant_bottleneck": self.cfg.latent_dim < 3,
            "train_scope": asdict(scope), "input_sha256": scope.input_sha256,
            "train_row_ids_sha256": scope.train_row_ids_sha256,
            "normalization": asdict(normalization) if normalization else None,
            "weights_sha256": weights_sha256(self.encoder), "encoder_file": "encoder.keras",
            "encoder_sha256": sha256_file(enc_path), "exported_without_decoder": True,
            "regimes": list(REGIMES), "operational_contract": "OPERATIONAL (inputs known at t; no target)",
        }
        man.update(self.extra_manifest())
        _atomic_json(os.path.join(out_dir, "donor_manifest.json"), man)
        return man


def make_extractor(family: str, cfg: ArchConfig, seed: int, **kw) -> UnivariateTemporalExtractor:
    if family not in FAMILIES:
        raise ContractError(f"unknown family {family!r}; declared: {sorted(FAMILIES)}")
    if FAMILIES[family].status not in RUNNABLE_STATUSES:
        raise SlotNotImplemented(f"{family}: {FAMILIES[family].status}")
    if FAMILIES[family].status == IMPLEMENTED_LANE_F:
        return LaneFExtractor(family, cfg, seed, **kw)
    return UnivariateTemporalExtractor(family, cfg, seed, **kw)


LANE_F_ARCHITECTURE_IDS = {"masked_temporal_ae": "lane_f_mtae_causal_attn_v1",
                           "past_to_current_siamese": "lane_f_p2c_causal_attn_v1"}
LANE_F_DEFAULTS = {"d_model": 32, "n_blocks": 2, "n_heads": 4, "dropout": 0.0,
                   "max_lag": None, "n_lineage": 3, "pairs_per_epoch": 512}


def batch_to_series(batch: TemporalBatch, period_seconds: int = 3600) -> Dict[str, np.ndarray]:
    """Contiguous (L, ch) arrays from right-edge windows with consecutive anchors (no subsampling)."""
    if len(batch) > 1 and np.any(np.diff(np.asarray(batch.anchor_ts, np.int64)) != period_seconds):
        raise ContractError("past_to_current_siamese needs consecutive anchors; do not subsample its windows")
    out = {}
    for k, a in batch.as_inputs().items():
        out[k] = np.concatenate([a[0], a[1:, -1, :]], axis=0) if len(batch) > 1 else a[0]
    return out


class LaneFExtractor(UnivariateTemporalExtractor):
    """Adapter over app.alt_extractor_families (lane F); one implementation, same interface and donors."""

    def __init__(self, family: str, cfg: ArchConfig, seed: int, learning_rate: float = 1e-3,
                 batch_size: int = 64, corruption: Optional[dict] = None, alt: Optional[dict] = None,
                 period_seconds: int = 3600):
        from app import alt_extractor_families as F
        self._F = F
        spec = FAMILIES[family]
        self.family, self.spec, self.cfg, self.seed = family, spec, cfg, int(seed)
        self.learning_rate, self.batch_size, self.period_seconds = float(learning_rate), int(batch_size), period_seconds
        self.corruption = corruption if corruption is not None else spec.default_corruption
        self.alt = dict(LANE_F_DEFAULTS, **(alt or {}))
        self.fit_report, self.decoder, self.training_model = None, None, None
        self.architecture_id, self.latent_dim = LANE_F_ARCHITECTURE_IDS[family], cfg.latent_dim
        self.enc_cfg = F.EncoderConfig(window=cfg.window, calendar_dims=cfg.calendar_dim, latent_dim=cfg.latent_dim,
                                       d_model=self.alt["d_model"], n_blocks=self.alt["n_blocks"],
                                       n_heads=self.alt["n_heads"], kernel_size=cfg.kernel_size,
                                       dropout=self.alt["dropout"])
        import keras
        keras.utils.set_random_seed(self.seed)
        self.encoder = F.build_causal_encoder(self.enc_cfg)  # untrained until fit
        assert_temporal_encoder(self.encoder, cfg.window)

    def extra_manifest(self) -> dict:
        from dataclasses import asdict as _asdict
        return {"alt_encoder_config": _asdict(self.enc_cfg), "alt_params": self.alt,
                "implementation": "app.alt_extractor_families (lane F clean-room; not a paper reproduction)",
                "per_instant_bottleneck": False}

    def reconstruct(self, batch):
        return None  # pretext reconstruction is defined on masked steps only; reported NOT_APPLICABLE

    def fit(self, train: TemporalBatch, val: TemporalBatch, es: EarlyStopConfig) -> dict:
        F, ratio = self._F, float(self.corruption["ratio"])
        t0 = time.perf_counter()
        common = dict(mask_ratio=ratio, epochs=es.max_epochs, patience=es.patience,
                      batch_size=self.batch_size, lr=self.learning_rate, seed=self.seed)
        if self.family == "masked_temporal_ae":
            enc, trainer, res = F.fit_mtae(self.enc_cfg, train.as_inputs(), val.as_inputs(), **common)
        else:
            enc, trainer, res = F.fit_p2c(self.enc_cfg, batch_to_series(train, self.period_seconds),
                                          batch_to_series(val, self.period_seconds), max_lag=self.alt["max_lag"],
                                          n_lineage=self.alt["n_lineage"],
                                          pairs_per_epoch=self.alt["pairs_per_epoch"], **common)
        if res.best_epoch < 0:
            raise TrainingDiverged("lane F fit produced no finite validation loss")
        self.encoder, self.training_model = enc, trainer
        n = len(res.history)
        self.fit_report = {
            "stop_reason": "patience" if n < es.max_epochs else "max_epochs", "epochs_run": n,
            "best_epoch": res.best_epoch, "best_val_loss": float(res.best_val),
            "val_loss_history": [h["val"] for h in res.history], "updates": int(res.updates),
            "fit_wall_seconds": time.perf_counter() - t0, "n_train_windows": len(train),
            "n_val_windows": len(val), "early_stop": asdict(es), "restored_best_checkpoint": True,
            "min_delta_applied": False, "monitor": "lane F masked_mse on fixed validation pretext masks",
        }
        return self.fit_report


_EXPECT_ALIASES = {"window": "window", "calendar_dim": "calendar_dim", "latent_dim": "latent_dim"}


def load_donor(donor_dir: str, regime: str, seed: Optional[int] = None, expected: Optional[dict] = None):
    """Verify identity, then return (encoder, record) under R0/R1/R2. Refuses before any fit."""
    if regime not in REGIMES:
        raise DonorIdentityError(f"regime must be one of {REGIMES}; got {regime!r}")
    mpath = os.path.join(donor_dir, "donor_manifest.json")
    if not os.path.isfile(mpath):
        raise DonorIdentityError(f"missing donor manifest at {mpath}")
    with open(mpath) as f:
        man = json.load(f)
    if man.get("schema") != DONOR_SCHEMA:
        raise DonorIdentityError(f"unknown donor schema {man.get('schema')!r}")
    for k, v in (expected or {}).items():
        if man.get(_EXPECT_ALIASES.get(k, k)) != v:
            raise DonorIdentityError(f"donor {k} = {man.get(k)!r}, expected {v!r}")
    enc_path = os.path.join(donor_dir, man.get("encoder_file", "encoder.keras"))
    if not os.path.isfile(enc_path):
        raise DonorIdentityError("missing donor encoder file")
    if sha256_file(enc_path) != man.get("encoder_sha256"):
        raise DonorIdentityError("encoder file digest mismatch")
    import keras
    try:
        enc = keras.saving.load_model(enc_path, compile=False)
    except Exception as e:  # corrupted archive
        raise DonorIdentityError(f"donor encoder unreadable: {e}") from e
    if weights_sha256(enc) != man.get("weights_sha256"):
        raise DonorIdentityError("donor weights digest mismatch")
    if tuple(enc.output.shape)[1] != man["window"]:
        raise DonorIdentityError("donor time grid differs from its manifest")
    assert_temporal_encoder(enc, man["window"])
    if regime == "R0":
        keras.utils.set_random_seed(int(man["seed"]) + 1 if seed is None else int(seed))
        if man["architecture_id"] in LANE_F_ARCHITECTURE_IDS.values():
            from app import alt_extractor_families as F
            enc = F.build_causal_encoder(F.EncoderConfig(**man["alt_encoder_config"]))
        else:
            enc = build_encoder(ArchConfig.from_dict(man["arch_config"]))
        enc.trainable = True
    else:
        enc.trainable = regime == "R2"
    rec = {"regime": regime, "donor_dir": os.path.abspath(donor_dir), "donor_weights_sha256": man["weights_sha256"],
           "initial_weights_sha256": weights_sha256(enc), "trainable": bool(enc.trainable),
           "architecture_id": man["architecture_id"], "feature_id": man.get("feature_id")}
    return enc, rec
