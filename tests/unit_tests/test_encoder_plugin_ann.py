import pytest
import numpy as np
from keras.models import Sequential
from app.plugins.encoder_plugin_ann import Plugin

@pytest.fixture
def encoder():
    return Plugin()

def test_set_params(encoder):
    encoder.set_params(input_dim=10, encoding_dim=5, epochs=20, batch_size=128)
    params = encoder.get_debug_info()
    assert params['input_dim'] == 10
    assert params['encoding_dim'] == 5
    assert params['epochs'] == 20
    assert params['batch_size'] == 128

def test_configure_size(encoder):
    encoder.configure_size(10, 5)
    assert isinstance(encoder.model, Sequential)
    assert encoder.model.layers[0].input_shape == (None, 10)
    assert encoder.model.layers[0].units == 5

def test_train(encoder):
    data = np.random.rand(100, 10)
    encoder.configure_size(10, 5)
    encoder.train(data)
    assert encoder.model is not None

def test_encode(encoder):
    data = np.random.rand(100, 10)
    encoder.configure_size(10, 5)
    encoder.train(data)
    encoded_data = encoder.encode(data)
    assert encoded_data.shape == (100, 5)

def test_save_and_load(encoder, tmpdir):
    model_path = tmpdir.join("test_model.h5")
    data = np.random.rand(100, 10)
    encoder.configure_size(10, 5)
    encoder.train(data)
    encoder.save(str(model_path))
    
    new_encoder = Plugin()
    new_encoder.load(str(model_path))
    assert new_encoder.model is not None
    assert new_encoder.model.layers[0].input_shape == (None, 10)
    assert new_encoder.model.layers[0].units == 5

def test_calculate_mse(encoder):
    original_data = np.random.rand(100, 10)
    reconstructed_data = np.random.rand(100, 10)
    mse = encoder.calculate_mse(original_data, reconstructed_data)
    assert isinstance(mse, float)
