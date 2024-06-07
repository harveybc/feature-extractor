import unittest
from unittest.mock import patch, MagicMock
from app.plugin_loader import load_plugin, load_encoder_decoder_plugins, get_plugin_params

class TestPluginLoader(unittest.TestCase):

    @patch('pkg_resources.get_entry_map')
    def test_load_plugin_success(self, mock_get_entry_map):
        # Setup
        plugin_name = 'default_encoder'
        plugin_group = 'feature_extractor.encoders'
        mock_plugin = MagicMock()
        mock_plugin.plugin_params = {'param1': 'value1'}
        mock_entry_point = MagicMock(load=MagicMock(return_value=mock_plugin))
        mock_get_entry_map.return_value = {plugin_name: mock_entry_point}

        # Execute
        plugin_class, required_params = load_plugin(plugin_group, plugin_name)

        # Verify
        mock_get_entry_map.assert_called_once_with('feature-extractor', plugin_group)
        self.assertEqual(plugin_class, mock_plugin)
        self.assertEqual(required_params, ['param1'])

    @patch('pkg_resources.get_entry_map')
    def test_load_plugin_not_found(self, mock_get_entry_map):
        # Setup
        plugin_name = 'non_existent_plugin'
        plugin_group = 'feature_extractor.encoders'
        mock_get_entry_map.return_value = {}

        # Execute & Verify
        with self.assertRaises(ImportError):
            load_plugin(plugin_group, plugin_name)

    @patch('pkg_resources.get_entry_map')
    def test_load_encoder_decoder_plugins(self, mock_get_entry_map):
        # Setup
        encoder_name = 'default_encoder'
        decoder_name = 'default_decoder'
        plugin_group_encoders = 'feature_extractor.encoders'
        plugin_group_decoders = 'feature_extractor.decoders'
        mock_encoder_plugin = MagicMock()
        mock_decoder_plugin = MagicMock()
        mock_encoder_plugin.plugin_params = {'param1': 'value1'}
        mock_decoder_plugin.plugin_params = {'param2': 'value2'}
        mock_encoder_entry_point = MagicMock(load=MagicMock(return_value=mock_encoder_plugin))
        mock_decoder_entry_point = MagicMock(load=MagicMock(return_value=mock_decoder_plugin))
        mock_get_entry_map.side_effect = [
            {encoder_name: mock_encoder_entry_point},
            {decoder_name: mock_decoder_entry_point}
        ]

        # Execute
        encoder_plugin, encoder_params, decoder_plugin, decoder_params = load_encoder_decoder_plugins(encoder_name, decoder_name)

        # Verify
        mock_get_entry_map.assert_any_call('feature-extractor', plugin_group_encoders)
        mock_get_entry_map.assert_any_call('feature-extractor', plugin_group_decoders)
        self.assertEqual(encoder_plugin, mock_encoder_plugin)
        self.assertEqual(encoder_params, ['param1'])
        self.assertEqual(decoder_plugin, mock_decoder_plugin)
        self.assertEqual(decoder_params, ['param2'])

    @patch('pkg_resources.get_entry_map')
    def test_get_plugin_params_success(self, mock_get_entry_map):
        # Setup
        plugin_name = 'default_encoder'
        plugin_group = 'feature_extractor.encoders'
        mock_plugin = MagicMock()
        mock_plugin.plugin_params = {'param1': 'value1'}
        mock_entry_point = MagicMock(load=MagicMock(return_value=mock_plugin))
        mock_get_entry_map.return_value = {plugin_name: mock_entry_point}

        # Execute
        params = get_plugin_params(plugin_group, plugin_name)

        # Verify
        mock_get_entry_map.assert_called_once_with('feature-extractor', plugin_group)
        self.assertEqual(params, {'param1': 'value1'})

    @patch('pkg_resources.get_entry_map')
    def test_get_plugin_params_not_found(self, mock_get_entry_map):
        # Setup
        plugin_name = 'non_existent_plugin'
        plugin_group = 'feature_extractor.encoders'
        mock_get_entry_map.return_value = {}

        # Execute
        params = get_plugin_params(plugin_group, plugin_name)

        # Verify
        mock_get_entry_map.assert_called_once_with('feature-extractor', plugin_group)
        self.assertEqual(params, {})

if __name__ == '__main__':
    unittest.main()
