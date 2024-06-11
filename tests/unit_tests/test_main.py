import pytest
from unittest.mock import patch, MagicMock
from app.main import main

@pytest.fixture
def mock_args():
    return [
        'script_name',
        'tests/data/csv_sel_unb_norm_512.csv',
        '--save_encoder', './encoder_ann.keras',
        '--save_decoder', './decoder_ann.keras'
    ]

@patch('app.main.parse_args')
@patch('app.main.load_config')
@patch('app.main.save_config')
@patch('app.main.process_data')
@patch('app.main.DEFAULT_VALUES', {
    'csv_input_path': './csv_input.csv',
    'csv_output_path': './csv_output.csv',
    'config_save_path': './config_out.json',
    'config_load_path': './config_in.json',
    'encoder_plugin': 'default',
    'decoder_plugin': 'default',
    'training_batch_size': 128,
    'epochs': 10,
    'plugin_directory': 'app/plugins/',
    'remote_log_url': None,
    'remote_config_url': None,
    'window_size': 128,
    'initial_encoding_dim': 4,
    'encoding_step_size': 4,
    'mse_threshold': 0.3,
    'quiet_mode': False,
    'remote_username': 'test',
    'remote_password': 'pass',
    'save_encoder_path': './encoder_ann.keras',
    'save_decoder_path': './decoder_ann.keras',
    'force_date': False,
    'headers': False,
    'incremental_search': True
})
def test_main(mock_process_data, mock_save_config, mock_load_config, mock_parse_args, mock_args):
    mock_parse_args.return_value = (MagicMock(load_config=None, save_config='config_out.json'), [])
    mock_process_data.return_value = (MagicMock(), {})
    mock_load_config.return_value = {}

    with patch('sys.argv', mock_args):
        main()

    mock_parse_args.assert_called_once()
    mock_load_config.assert_not_called()
    mock_save_config.assert_called_once()
    mock_process_data.assert_called_once()

@patch('app.main.parse_args')
@patch('app.main.load_config')
@patch('app.main.save_config')
@patch('app.main.process_data')
@patch('app.main.DEFAULT_VALUES', {
    'csv_input_path': './csv_input.csv',
    'csv_output_path': './csv_output.csv',
    'config_save_path': './config_out.json',
    'config_load_path': './config_in.json',
    'encoder_plugin': 'default',
    'decoder_plugin': 'default',
    'training_batch_size': 128,
    'epochs': 10,
    'plugin_directory': 'app/plugins/',
    'remote_log_url': None,
    'remote_config_url': None,
    'window_size': 128,
    'initial_encoding_dim': 4,
    'encoding_step_size': 4,
    'mse_threshold': 0.3,
    'quiet_mode': False,
    'remote_username': 'test',
    'remote_password': 'pass',
    'save_encoder_path': './encoder_ann.keras',
    'save_decoder_path': './decoder_ann.keras',
    'force_date': False,
    'headers': False,
    'incremental_search': True
})
def test_main_with_invalid_range(mock_process_data, mock_save_config, mock_load_config, mock_parse_args, mock_args):
    mock_parse_args.return_value = (MagicMock(), ['--range', '(invalid)'])
    mock_process_data.return_value = (MagicMock(), {})
    mock_load_config.return_value = {}

    with patch('sys.argv', mock_args):
        with patch('sys.stderr', new_callable=MagicMock()) as mock_stderr:
            main()

        mock_parse_args.assert_called_once()
        mock_load_config.assert_not_called()
        mock_save_config.assert_not_called()
        mock_process_data.assert_not_called()
        mock_stderr.write.assert_any_call('Error: Invalid format for --range argument\n')

if __name__ == "__main__":
    pytest.main()
