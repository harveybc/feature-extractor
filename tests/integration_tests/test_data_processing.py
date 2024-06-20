import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.data_processor import train_autoencoder
from app.autoencoder_manager import AutoencoderManager

@pytest.fixture
def autoencoder_manager():
    return MagicMock(spec=AutoencoderManager)

@pytest.fixture
def mock_data():
    return pd.DataFrame(np.random.random((1000, 10)))

@pytest.fixture
def config():
    return {
        'mse_threshold': 0.01,
        'initial_encoding_dim': 4,
        'encoding_step_size': 4,
        'incremental_search': True,
        'epochs': 10,
        'csv_file': 'path/to/mock_csv.csv',
        'headers': True,
        'force_date': False
    }

def test_train_autoencoder(autoencoder_manager, mock_data, config):
    with patch('app.data_processor.AutoencoderManager', return_value=autoencoder_manager):
        trained_manager = train_autoencoder(autoencoder_manager, mock_data.values, config['mse_threshold'], config['initial_encoding_dim'], config['encoding_step_size'], config['incremental_search'], config['epochs'])
        assert trained_manager is not None
        print("Checking if build_autoencoder was called...")
        autoencoder_manager.build_autoencoder.assert_called()
        print("build_autoencoder was called.")
