import pytest
import numpy as np
import pandas as pd
from app.data_processor import train_autoencoder, process_data
from app.autoencoder_manager import AutoencoderManager
from unittest.mock import MagicMock

@pytest.fixture
def config():
    return {
        'csv_file': 'path/to/mock_csv.csv',
        'headers': False,
        'force_date': False,
        'window_size': 128,
        'mse_threshold': 0.005,
        'initial_encoding_dim': 4,
        'encoding_step_size': 4,
        'epochs': 10,
        'incremental_search': True
    }

@pytest.fixture
def mock_data():
    return pd.DataFrame(np.random.random((1000, 10)))

@pytest.fixture
def autoencoder_manager():
    mock_autoencoder_manager = MagicMock(spec=AutoencoderManager)
    mock_autoencoder_manager.encode_data.return_value = np.random.random((872, 4))
    mock_autoencoder_manager.decode_data.return_value = np.random.random((872, 128))
    mock_autoencoder_manager.calculate_mse.return_value = 0.004
    return mock_autoencoder_manager

def test_train_autoencoder(autoencoder_manager, mock_data, config):
    trained_manager = train_autoencoder(autoencoder_manager, mock_data.values, config['mse_threshold'], config['initial_encoding_dim'], config['encoding_step_size'], config['incremental_search'], config['epochs'])
    assert trained_manager is not None
    autoencoder_manager.build_autoencoder.assert_called()
    autoencoder_manager.train_autoencoder.assert_called()

def test_process_data(config, mock_data):
    # Mock the load_csv function to return mock_data
    from app.data_handler import load_csv
    load_csv_mock = MagicMock(return_value=mock_data)
    pd.read_csv = load_csv_mock

    processed_data, debug_info = process_data(config)
    assert processed_data is not None
    assert isinstance(processed_data, dict)
    assert len(processed_data) == mock_data.shape[1]
    for column, data in processed_data.items():
        assert data.shape[1] == config['window_size']  # Check if windowed data has correct shape

if __name__ == "__main__":
    pytest.main()
