import pytest
from app.plugin_loader import load_plugin, load_encoder_decoder_plugins, get_plugin_params
from app.plugins.encoder_plugin_ann import Plugin as EncoderPlugin

def test_load_plugin_success():
    plugin_class, required_params = load_plugin('feature_extractor.encoders', 'encoder_plugin_ann')
    assert plugin_class.plugin_params == {'epochs': 10, 'batch_size': 256}
    assert required_params == ['epochs', 'batch_size']

def test_load_plugin_general_exception():
    with pytest.raises(Exception) as excinfo:
        load_plugin('feature_extractor.encoders', 'non_existent_plugin')
    assert 'Plugin non_existent_plugin not found in group feature_extractor.encoders' in str(excinfo.value)

def test_load_encoder_decoder_plugins():
    encoder_plugin, encoder_params, decoder_plugin, decoder_params = load_encoder_decoder_plugins('encoder_plugin_ann', 'decoder_plugin_ann')
    assert encoder_plugin.plugin_params == {'epochs': 10, 'batch_size': 256}
    assert encoder_params == ['epochs', 'batch_size']
    assert decoder_plugin.plugin_params == {'param2': 'value2'}
    assert decoder_params == ['param2']

def test_get_plugin_params_success():
    params = get_plugin_params('feature_extractor.encoders', 'encoder_plugin_ann')
    assert params == {'epochs': 10, 'batch_size': 256}

def test_get_plugin_params_key_error():
    params = get_plugin_params('feature_extractor.encoders', 'non_existent_plugin')
    assert params == {}

def test_get_plugin_params_general_exception():
    with pytest.raises(Exception) as excinfo:
        get_plugin_params('feature_extractor.encoders', 'non_existent_plugin')
    assert 'Plugin non_existent_plugin not found in group feature_extractor.encoders' in str(excinfo.value)

if __name__ == '__main__':
    pytest.main()
