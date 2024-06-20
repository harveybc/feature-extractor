import pytest
from app.plugin_loader import load_encoder_decoder_plugins

def test_load_default_plugins():
    encoder_plugin, encoder_params, decoder_plugin, decoder_params = load_encoder_decoder_plugins('default', 'default')
    assert encoder_plugin.plugin_params == {'epochs': 10, 'batch_size': 256}
    assert decoder_plugin.plugin_params == {'epochs': 10, 'batch_size': 256}

def test_load_specific_plugins():
    encoder_plugin, encoder_params, decoder_plugin, decoder_params = load_encoder_decoder_plugins('ann', 'cnn')
    assert encoder_plugin.plugin_params == {'epochs': 10, 'batch_size': 256}  # Adjust based on actual plugin params
    assert decoder_plugin.plugin_params == {'epochs': 10, 'batch_size': 256}  # Adjust based on actual plugin params

def test_load_non_existent_plugins():
    with pytest.raises(ImportError):
        load_encoder_decoder_plugins('non_existent_encoder', 'non_existent_decoder')
