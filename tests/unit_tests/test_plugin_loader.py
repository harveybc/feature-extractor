import pytest
from unittest.mock import patch, MagicMock
from app.plugin_loader import load_plugin, load_encoder_decoder_plugins, get_plugin_params

def test_load_plugin_success():
    with patch('app.plugin_loader.pkg_resources.get_entry_map') as mock_get_entry_map:
        mock_entry_point = MagicMock()
        mock_entry_point.load.return_value.plugin_params = {'param1': 'value1'}
        mock_get_entry_map.return_value = {'test_plugin': mock_entry_point}

        plugin_class, required_params = load_plugin('test_group', 'test_plugin')
        assert plugin_class is not None
        assert required_params == ['param1']

def test_load_plugin_key_error():
    with patch('app.plugin_loader.pkg_resources.get_entry_map') as mock_get_entry_map:
        mock_get_entry_map.return_value = {}

        with pytest.raises(ImportError):
            load_plugin('test_group', 'non_existent_plugin')

def test_load_plugin_exception():
    with patch('app.plugin_loader.pkg_resources.get_entry_map') as mock_get_entry_map:
        mock_get_entry_map.side_effect = Exception('Unexpected error')

        with pytest.raises(Exception):
            load_plugin('test_group', 'test_plugin')

def test_load_encoder_decoder_plugins():
    with patch('app.plugin_loader.load_plugin') as mock_load_plugin:
        mock_load_plugin.side_effect = [
            (MagicMock(), ['param1']),
            (MagicMock(), ['param2'])
        ]

        encoder_plugin, encoder_params, decoder_plugin, decoder_params = load_encoder_decoder_plugins('encoder', 'decoder')
        assert encoder_plugin is not None
        assert encoder_params == ['param1']
        assert decoder_plugin is not None
        assert decoder_params == ['param2']

def test_get_plugin_params_success():
    with patch('app.plugin_loader.pkg_resources.get_entry_map') as mock_get_entry_map:
        mock_entry_point = MagicMock()
        mock_entry_point.load.return_value.plugin_params = {'param1': 'value1'}
        mock_get_entry_map.return_value = {'test_plugin': mock_entry_point}

        params = get_plugin_params('test_group', 'test_plugin')
        assert params == {'param1': 'value1'}

def test_get_plugin_params_key_error():
    with patch('app.plugin_loader.pkg_resources.get_entry_map') as mock_get_entry_map:
        mock_get_entry_map.return_value = {}

        params = get_plugin_params('test_group', 'non_existent_plugin')
        assert params == {}

def test_get_plugin_params_exception():
    with patch('app.plugin_loader.pkg_resources.get_entry_map') as mock_get_entry_map:
        mock_get_entry_map.side_effect = Exception('Unexpected error')

        params = get_plugin_params('test_group', 'test_plugin')
        assert params == {}

if __name__ == "__main__":
    pytest.main()
