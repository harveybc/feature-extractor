"""WP25: the `rnn` encoder entry point builds and encodes, on CPU, with no training.

These tests exist because `setup.py` declared `rnn` for a module that was not in the checkout: a declared
plugin that cannot be built is a wrong option for anything that reads the registry. They build a tiny model
and push four small windows through it; the weights are untrained, so this checks wiring and shapes, never
reconstruction quality.
"""

import math
import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from app.plugins.decoder_plugin_rnn import Plugin as Decoder
from app.plugins.encoder_plugin_rnn import Plugin as Encoder

WINDOW, CHANNELS, LATENT, BATCH = 8, 3, 4, 4


@pytest.fixture
def windows():
    rng = np.random.default_rng(20260925)
    return rng.standard_normal((BATCH, WINDOW, CHANNELS)).astype("float32")


@pytest.fixture
def encoder():
    plugin = Encoder()
    plugin.set_params(initial_layer_size=8, layer_size_divisor=2)
    return plugin


def _latent_length(window):
    return math.ceil(math.ceil(window / 2) / 2)


def test_plugin_params_shape_matches_the_sibling_encoders(encoder):
    for key in ("activation", "intermediate_layers", "learning_rate", "dropout_rate",
                "initial_layer_size", "layer_size_divisor", "l2_reg"):
        assert key in Encoder.plugin_params
    assert Encoder.plugin_params["rnn_type"] == "simple_rnn"


def test_set_params_and_debug_info(encoder):
    encoder.set_params(intermediate_layers=3)
    assert encoder.params["intermediate_layers"] == 3
    encoder.configure_size(WINDOW, LATENT, CHANNELS, True, config={"window_size": WINDOW})
    debug_info = {}
    encoder.add_debug_info(debug_info)
    assert debug_info["input_shape"] == WINDOW
    assert debug_info["rnn_type"] == "simple_rnn"


def test_configure_size_builds_a_compiled_model(encoder):
    encoder.configure_size(WINDOW, LATENT, CHANNELS, True, config={"window_size": WINDOW})
    assert encoder.encoder_model is not None
    assert encoder.encoder_model.input_shape == (None, WINDOW, CHANNELS)
    assert encoder.encoder_model.output_shape[1] == _latent_length(WINDOW)


def test_encode_a_tiny_window(encoder, windows):
    encoder.configure_size(WINDOW, LATENT, CHANNELS, True, config={"window_size": WINDOW})
    encoded = encoder.encode(windows)
    assert encoded.shape[0] == BATCH
    assert encoded.shape[1] == _latent_length(WINDOW)
    assert np.isfinite(encoded).all()


def test_gru_variant_builds_and_encodes(encoder, windows):
    encoder.set_params(rnn_type="gru")
    encoder.configure_size(WINDOW, LATENT, CHANNELS, True, config={"window_size": WINDOW})
    encoded = encoder.encode(windows)
    assert encoded.shape[:2] == (BATCH, _latent_length(WINDOW))


def test_unknown_rnn_type_is_refused_by_name(encoder):
    encoder.set_params(rnn_type="quantum")
    with pytest.raises(ValueError, match="quantum"):
        encoder.configure_size(WINDOW, LATENT, CHANNELS, True, config={"window_size": WINDOW})


def test_save_and_load_round_trip(encoder, windows, tmp_path):
    encoder.configure_size(WINDOW, LATENT, CHANNELS, True, config={"window_size": WINDOW})
    before = encoder.encode(windows)
    path = tmp_path / "encoder_rnn.keras"
    encoder.save(str(path))
    reloaded = Encoder()
    reloaded.load(str(path))
    np.testing.assert_allclose(before, reloaded.encode(windows), rtol=1e-5, atol=1e-6)


def test_decoder_returns_the_window_shape(encoder, windows):
    encoder.configure_size(WINDOW, LATENT, CHANNELS, True, config={"window_size": WINDOW})
    encoded = encoder.encode(windows)
    decoder = Decoder()
    decoder.configure_size(LATENT, WINDOW, CHANNELS, encoded.shape[1:], True, config={"window_size": WINDOW})
    decoded = decoder.decode(encoded)
    assert decoded.shape == (BATCH, WINDOW, CHANNELS)
