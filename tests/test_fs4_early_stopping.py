"""Focused CPU tests for FS4's paired TRAIN reconstruction monitor."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

from app import fs4_extractibility as X
from app import univariate_temporal as U


class _Iterations:
    def __init__(self):
        self.value = 0

    def numpy(self):
        return self.value


class _Training:
    def __init__(self):
        self.epoch = -1
        self.weight = -1
        self.optimizer = type("Optimizer", (), {"iterations": _Iterations()})()

    def fit(self, *args, **kwargs):
        self.epoch += 1
        self.weight = self.epoch
        self.optimizer.iterations.value += 1

    def get_weights(self):
        return [np.array([self.weight])]

    def set_weights(self, weights):
        self.weight = int(weights[0][0])


def _batch(anchor):
    return U.TemporalBatch.from_inputs({
        "signal": np.zeros((1, 4, 1), np.float32),
        "observed_mask": np.ones((1, 4, 1), np.float32),
        "delta_time": np.zeros((1, 4, 1), np.float32),
        "calendar": np.zeros((1, 4, 1), np.float32),
    }, anchor_ts=np.array([anchor]))


def _run(monkeypatch, fit_losses, es_losses, patience=2):
    model = _Training()
    masks = []

    def mask(observed, seed, fraction):
        masks.append(seed)
        return np.array([[True, False, False, False]])

    def reconstruct(training, batch):
        loss = (fit_losses if batch.anchor_ts[0] == 1 else es_losses)[training.epoch]
        return np.full((1, 4, 1), np.sqrt(loss), np.float32)

    monkeypatch.setattr(X, "hidden_mask", mask)
    monkeypatch.setattr(X, "reconstruct", reconstruct)
    monkeypatch.setattr(X, "weights_digest", lambda models: str(model.weight))
    report = X.fit_trained(model, object(), object(), _batch(1), _batch(2),
                           X.Hyper(max_epochs=len(fit_losses), patience=patience), 3,
                           np.array([[True, False, False, False]]))
    return model, report, masks


def test_patience_follows_mean_and_restores_its_checkpoint(monkeypatch):
    # Validation alone prefers epoch 0; the paired mean improves again at epoch 2.
    model, report, masks = _run(monkeypatch, [8, 5, 2, 3, 4], [2, 4, 5, 6, 7])
    assert report["stop_reason"] == "patience"
    assert report["epochs_run"] == 5
    assert report["chosen_epoch"] == model.weight == 2
    assert report["best_monitor_hidden_mse"] == pytest.approx(3.5)
    assert report["fit_hidden_mse_history"] == pytest.approx([8, 5, 2, 3, 4])
    assert report["es_hidden_mse_history"] == pytest.approx([2, 4, 5, 6, 7])
    assert report["monitor_hidden_mse_history"] == pytest.approx([5, 4.5, 3.5, 4.5, 5.5])
    assert report["chosen_weights_sha256"] == "2"
    assert report["final_weights_sha256"] == "4"
    assert report["restored_best_checkpoint"] is True
    assert report["best_es_hidden_mse"] == pytest.approx(2)
    assert report["selected_es_hidden_mse"] == pytest.approx(5)
    assert report["min_es_hidden_mse"] == pytest.approx(2)
    assert report["es_degradation_from_min"] == pytest.approx(3)
    assert report["fit_monitor_hidden_points"] == report["es_hidden_points"] == 1
    assert len(masks) == 6  # one fixed fit monitor mask, plus one fit update mask per epoch


@pytest.mark.parametrize("bad_side", ["fit", "es"])
def test_nonfinite_monitor_refuses_even_with_prior_checkpoint(monkeypatch, bad_side):
    fit = [1, float("nan")] if bad_side == "fit" else [1, 1]
    es = [1, 1] if bad_side == "fit" else [1, float("inf")]
    with pytest.raises(U.TrainingDiverged, match="non-finite"):
        _run(monkeypatch, fit, es)
