import pytest
import pandas as pd
import numpy as np
from app.data_processor import train_autoencoder
from app.autoencoder_manager import AutoencoderManager
from app.plugin_loader import load_plugin

@pytest.fixture
def mock_data():
    return pd.DataFrame(np.random.random((1000, 10)))

@pytest.fixture
def config():
    return {
        'csv_file': 'path/to/mock_csv.csv',
        'encoding_step_size': 4,
        'epochs': 10,
        'force_date': False,
        'mse_threshold': 0.01,
        'initial_encoding_dim': 4,
        'incremental_search': True
    }

def test_train_autoencoder(mock_data, config):
    encoder_plugin, encoder_params = load_plugin('feature_extractor.encoders', 'default')
    decoder_plugin, decoder_params = load_plugin('feature_extractor.decoders', 'default')
    
    autoencoder_manager = AutoencoderManager(input_dim=mock_data.shape[1], encoding_dim=config['initial_encoding_dim'])
    
    trained_manager = train_autoencoder(autoencoder_manager, mock_data.values, config['mse_threshold'], config['initial_encoding_dim'], config['encoding_step_size'], config['incremental_search'], config['epochs'])
    assert trained_manager is not None
    print("Checking if build_autoencoder was called...")
    assert autoencoder_manager.autoencoder_model is not None
    print("Autoencoder model was built.")
