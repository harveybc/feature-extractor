import pytest
from unittest.mock import patch, MagicMock
from importlib.metadata import EntryPoint
from app.plugin_loader import load_plugin, load_encoder_decoder_plugins, get_plugin_params

def test_load_plugin_success():
    mock_entry_point = EntryPoint(name='mock_plugin', value='mock.module:Plugin', group='feature_extractor.encoders')
    mock_plugin_class = MagicMock(plugin_params={'param1': 'value1'})
    mock_entry_point.load = MagicMock(return_value=mock_plugin_class)

    with patch('importlib.metadata.entry_points', return_value={'feature_extractor.encoders': [mock_entry_point]}):
        plugin_class, required_params = load_plugin('feature_extractor.encoders', 'mock_plugin')
        assert plugin_class.plugin_params == {'param1': 'value1'}
        assert required_params == ['param1']

def test_load_plugin_key_error():
    with patch('importlib.metadata.entry_points', return_value={'feature_extractor.encoders': []}):
        with pytest.raises(ImportError):
            load_plugin('feature_extractor.encoders', 'mock_plugin')

def test_load_plugin_general_exception():
    with patch('importlib.metadata.entry_points', side_effect=Exception('General error')):
        with pytest.raises(Exception) as excinfo:
            load_plugin('feature_extractor.encoders', 'mock_plugin')
        assert 'General error' in str(excinfo.value)

def test_load_encoder_decoder_plugins():
    mock_encoder_entry_point = EntryPoint(name='mock_encoder', value='mock.module:EncoderPlugin', group='feature_extractor.encoders')
    mock_decoder_entry_point = EntryPoint(name='mock_decoder', value='mock.module:DecoderPlugin', group='feature_extractor.decoders')
    mock_encoder_class = MagicMock(plugin_params={'param1': 'value1'})
    mock_decoder_class = MagicMock(plugin_params={'param2': 'value2'})
    mock_encoder_entry_point.load = MagicMock(return_value=mock_encoder_class)
    mock_decoder_entry_point.load = MagicMock(return_value=mock_decoder_class)

    with patch('importlib.metadata.entry_points', return_value={
        'feature_extractor.encoders': [mock_encoder_entry_point],
        'feature_extractor.decoders': [mock_decoder_entry_point]
    }):
        encoder_plugin, encoder_params, decoder_plugin, decoder_params = load_encoder_decoder_plugins('mock_encoder', 'mock_decoder')
        assert encoder_plugin.plugin_params == {'param1': 'value1'}
        assert encoder_params == ['param1']
        assert decoder_plugin.plugin_params == {'param2': 'value2'}
        assert decoder_params == ['param2']

def test_get_plugin_params_success():
    mock_entry_point = EntryPoint(name='mock_plugin', value='mock.module:Plugin', group='feature_extractor.encoders')
    mock_plugin_class = MagicMock(plugin_params={'param1': 'value1'})
    mock_entry_point.load = MagicMock(return_value=mock_plugin_class)

    with patch('importlib.metadata.entry_points', return_value={'feature_extractor.encoders': [mock_entry_point]}):
        params = get_plugin_params('feature_extractor.encoders', 'mock_plugin')
        assert params == {'param1': 'value1'}

def test_get_plugin_params_key_error():
    with patch('importlib.metadata.entry_points', return_value={'feature_extractor.encoders': []}):
        params = get_plugin_params('feature_extractor.encoders', 'mock_plugin')
        assert params == {}

def test_get_plugin_params_general_exception():
    with patch('importlib.metadata.entry_points', side_effect=Exception('General error')):
        params = get_plugin_params('feature_extractor.encoders', 'mock_plugin')
        assert params == {}
