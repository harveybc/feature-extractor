import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from app.plugins.encoder_plugin_ann import Plugin

@pytest.fixture
def sample_data():
    return np.random.rand(100, 10)

@pytest.fixture
def encoder():
    return Plugin()

def test_configure_size(encoder):
    encoder.configure_size(10, 256)
    assert isinstance(encoder.model, Sequential)

def test_train(encoder, sample_data):
    encoder.configure_size(10, 256)
    encoder.train(sample_data)
    assert encoder.model is not None

def test_encode(encoder, sample_data):
    encoder.configure_size(10, 256)
    encoder.train(sample_data)
    encoded_data = encoder.encode(sample_data)
    assert encoded_data.shape[1] == 256

def test_calculate_mse(encoder, sample_data):
    encoder.configure_size(10, 256)
    encoder.train(sample_data)
    encoded_data = encoder.encode(sample_data)
    decoded_data = encoder.model.predict(encoded_data)
    mse = encoder.calculate_mse(sample_data, decoded_data)
    assert mse >= 0

@patch("keras.models.load_model")
@patch("keras.models.save_model")
def test_save_and_load(mock_save_model, mock_load_model, encoder):
    encoder.configure_size(10, 256)
    encoder.train(np.random.rand(100, 10))
    encoder.save("test_model.h5")
    mock_save_model.assert_called_once_with("test_model.h5")
    encoder.load("test_model.h5")
    mock_load_model.assert_called_once_with("test_model.h5")
