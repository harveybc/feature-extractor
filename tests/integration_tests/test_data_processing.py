import pytest
from app.data_processor import train_autoencoder
from app.autoencoder_manager import AutoencoderManager
from app.plugin_loader import load_plugin

@pytest.fixture
def mock_data():
    return pd.DataFrame(np.random.rand(1000, 10))

@pytest.fixture
def config():
    return {
        'csv_file': 'path/to/mock_csv.csv',
        'mse_threshold': 0.005,
        'initial_encoding_dim': 4,
        'encoding_step_size': 4,
        'incremental_search': True,
        'epochs': 10,
        'force_date': False,
        'headers': False,
    }

def test_train_autoencoder(mock_data, config):
    encoder_plugin, encoder_params = load_plugin('feature_extractor.encoders', 'default')
    decoder_plugin, decoder_params = load_plugin('feature_extractor.decoders', 'default')

    autoencoder_manager = AutoencoderManager(input_dim=mock_data.shape[1], encoding_dim=config['initial_encoding_dim'])

    trained_manager = train_autoencoder(autoencoder_manager, mock_data.values, config['mse_threshold'], config['initial_encoding_dim'], config['encoding_step_size'], config['incremental_search'], config['epochs'])
    assert trained_manager is not None
    print("Checking if build_autoencoder was called...")
    assert autoencoder_manager.autoencoder_model is not None
