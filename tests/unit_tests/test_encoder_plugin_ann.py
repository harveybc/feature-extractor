import pytest
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from unittest.mock import patch, MagicMock
from app.plugins.encoder_plugin_ann import Plugin

# Fixtures for setting up mock data and configurations
@pytest.fixture
def mock_data():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

@pytest.fixture
def encoder_plugin():
    return Plugin()

def test_set_params(encoder_plugin):
    encoder_plugin.set_params(input_dim=5, encoding_dim=2, epochs=15, batch_size=32)
    assert encoder_plugin.params['input_dim'] == 5
    assert encoder_plugin.params['encoding_dim'] == 2
    assert encoder_plugin.params['epochs'] == 15
    assert encoder_plugin.params['batch_size'] == 32

def test_get_debug_info(encoder_plugin):
    encoder_plugin.set_params(input_dim=5, encoding_dim=2)
    debug_info = encoder_plugin.get_debug_info()
    assert debug_info['input_dim'] == 5
    assert debug_info['encoding_dim'] == 2
    assert debug_info['epochs'] == 10  # default value
    assert debug_info['batch_size'] == 256  # default value

def test_add_debug_info(encoder_plugin):
    encoder_plugin.set_params(input_dim=5, encoding_dim=2)
    debug_info = {}
    encoder_plugin.add_debug_info(debug_info)
    assert debug_info['input_dim'] == 5
    assert debug_info['encoding_dim'] == 2
    assert debug_info['epochs'] == 10  # default value
    assert debug_info['batch_size'] == 256  # default value

def test_configure_size(encoder_plugin):
    encoder_plugin.configure_size(input_dim=3, encoding_dim=2)
    assert encoder_plugin.params['input_dim'] == 3
    assert encoder_plugin.params['encoding_dim'] == 2
    assert encoder_plugin.model is not None
    assert encoder_plugin.encoder_model is not None

def test_train(encoder_plugin, mock_data):
    encoder_plugin.configure_size(input_dim=3, encoding_dim=2)
    with patch.object(encoder_plugin.model, 'fit') as mock_fit:
        encoder_plugin.train(mock_data)
        mock_fit.assert_called_once_with(mock_data, mock_data, epochs=10, batch_size=256, verbose=1)

def test_encode(encoder_plugin, mock_data):
    encoder_plugin.configure_size(input_dim=3, encoding_dim=2)
    with patch.object(encoder_plugin.encoder_model, 'predict') as mock_predict:
        encoder_plugin.encode(mock_data)
        mock_predict.assert_called_once_with(mock_data)

def test_save(encoder_plugin):
    encoder_plugin.configure_size(input_dim=3, encoding_dim=2)
    with patch('app.plugins.encoder_plugin_ann.save_model') as mock_save_model:
        encoder_plugin.save('test_path')
        mock_save_model.assert_called_once_with(encoder_plugin.model, 'test_path')

def test_load(encoder_plugin):
    with patch('app.plugins.encoder_plugin_ann.load_model') as mock_load_model:
        encoder_plugin.load('test_path')
        mock_load_model.assert_called_once_with('test_path')
        assert encoder_plugin.encoder_model is not None

def test_calculate_mse(encoder_plugin, mock_data):
    encoder_plugin.configure_size(input_dim=3, encoding_dim=2)
    encoder_plugin.train(mock_data)
    reconstructed_data = encoder_plugin.model.predict(mock_data)
    mse = encoder_plugin.calculate_mse(mock_data, reconstructed_data)
    assert mse >= 0

if __name__ == "__main__":
    pytest.main()
