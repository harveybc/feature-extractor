import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.plugins.encoder_plugin_ann import Plugin

def test_configure_size():
    plugin = Plugin()
    plugin.configure_size(input_dim=10, encoding_dim=5)
    assert plugin.model.input_shape == (None, 10)
    assert plugin.model.output_shape == (None, 10)

def test_train():
    plugin = Plugin()
    data = np.random.rand(100, 10)
    plugin.configure_size(input_dim=10, encoding_dim=5)
    plugin.train(data)
    assert plugin.model is not None

def test_encode():
    plugin = Plugin()
    data = np.random.rand(10, 10)
    plugin.configure_size(input_dim=10, encoding_dim=5)
    plugin.train(data)
    encoded_data = plugin.encode(data)
    assert encoded_data.shape[1] == 5

def test_calculate_mse():
    plugin = Plugin()
    data = np.random.rand(10, 10)
    decoded_data = data  # Mock the decoded data as same as input for simplicity
    mse = plugin.calculate_mse(data, decoded_data)
    assert mse == 0

@patch("keras.models.save_model")
@patch("keras.models.load_model")
def test_save_and_load(mock_save_model, mock_load_model):
    plugin = Plugin()
    plugin.configure_size(input_dim=10, encoding_dim=5)
    plugin.save("test_model.keras")
    mock_save_model.assert_called_once()
    plugin.load("test_model.keras")
    mock_load_model.assert_called_once()

if __name__ == "__main__":
    pytest.main()
