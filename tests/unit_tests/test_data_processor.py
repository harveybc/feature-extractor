import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from app.data_processor import train_autoencoder, process_data
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Sample data for tests
sample_data = pd.DataFrame({
    '0': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

# Mock Encoder and Decoder classes
class MockEncoder:
    def __init__(self):
        self.model = MagicMock()
        self.model.input_shape = (None, 512)
        self.model.output_shape = (None, 4)

    def configure_size(self, input_dim, encoding_dim):
        pass

    def train(self, data):
        pass

    def encode(self, data):
        return data

    def save(self, path):
        pass

    def calculate_mse(self, original_data, reconstructed_data):
        return mean_squared_error(original_data, reconstructed_data)

class MockDecoder:
    def __init__(self):
        self.model = MagicMock()
        self.model.input_shape = (None, 4)
        self.model.output_shape = (None, 512)

    def configure_size(self, encoding_dim, output_dim):
        pass

    def train(self, encoded_data, original_data):
        pass

    def decode(self, encoded_data):
        return encoded_data

    def save(self, path):
        pass

@pytest.fixture
def mock_config():
    return {
        'csv_file': 'tests/data/csv_sel_unb_norm_512.csv',
        'headers': False,
        'force_date': False,
        'encoder_plugin': 'mock_encoder',
        'decoder_plugin': 'mock_decoder',
        'window_size': 3,
        'mse_threshold': 0.3,
        'initial_encoding_dim': 4,
        'encoding_step_size': 2,
        'incremental_search': False,
        'epochs': 10,
        'save_encoder_path': './encoder',
        'save_decoder_path': './decoder',
        'csv_output_path': './output'
    }

@patch('app.plugin_loader.load_plugin')
@patch('app.data_handler.load_csv')
@patch('app.data_handler.write_csv')
@patch('app.reconstruction.unwindow_data')
def test_process_data(mock_unwindow_data, mock_write_csv, mock_load_csv, mock_load_plugin, mock_config):
    mock_load_csv.return_value = sample_data
    mock_load_plugin.side_effect = [(MockEncoder, []), (MockDecoder, [])]
    mock_unwindow_data.return_value = pd.DataFrame({'Output': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    reconstructed_data, debug_info = process_data(mock_config)

    assert isinstance(reconstructed_data, pd.DataFrame)
    assert 'mean_squared_error_0' in debug_info
    assert 'mean_absolute_error_0' in debug_info

def test_train_autoencoder():
    encoder = MockEncoder()
    decoder = MockDecoder()
    data = np.array([[1, 2, 3], [4, 5, 6
