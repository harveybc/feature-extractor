"""The versioned I5-P encoder must actually use the origin observation."""

import numpy as np

from app import fs4_extractibility as X


def _positive_weights(models):
    for model in models:
        model.set_weights([
            np.full(tuple(weight.shape), 0.01 if len(weight.shape) > 1 else 0.1, dtype="float32")
            for weight in model.weights
        ])


def _inputs():
    return {
        "signal": np.ones((1, 24, 1), dtype="float32"),
        "observed_mask": np.ones((1, 24, 1), dtype="float32"),
        "delta_time": np.ones((1, 24, 1), dtype="float32"),
        "calendar": np.ones((1, 24, 4), dtype="float32"),
    }


def test_origin_covering_builder_preserves_shape_and_reaches_current_row():
    hp = X.Hyper()
    old_encoder, old_decoder, old_model = X.build_models(hp, calendar_dim=4)
    new_encoder, new_decoder, new_model = X.build_origin_covering_models(hp, calendar_dim=4)
    assert tuple(new_encoder.output.shape[1:]) == (6, 8)
    assert tuple(new_model.output.shape[1:]) == (24, 1)
    _positive_weights([old_encoder, old_decoder])
    _positive_weights([new_encoder, new_decoder])

    before = _inputs()
    after = {name: value.copy() for name, value in before.items()}
    after["signal"][0, 23, 0] += 10.0

    old_latent_before = np.asarray(old_encoder(before, training=False))
    old_latent_after = np.asarray(old_encoder(after, training=False))
    new_latent_before = np.asarray(new_encoder(before, training=False))
    new_latent_after = np.asarray(new_encoder(after, training=False))
    np.testing.assert_array_equal(old_latent_before, old_latent_after)
    assert np.any(new_latent_after[:, -1] != new_latent_before[:, -1])

    old_recon_before = np.asarray(old_model(before, training=False))
    old_recon_after = np.asarray(old_model(after, training=False))
    new_recon_before = np.asarray(new_model(before, training=False))
    new_recon_after = np.asarray(new_model(after, training=False))
    np.testing.assert_array_equal(old_recon_before[:, -1], old_recon_after[:, -1])
    assert np.any(new_recon_after[:, -1] != new_recon_before[:, -1])
