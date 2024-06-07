import pytest
from unittest.mock import patch, MagicMock
import json
import numpy as np
import pandas as pd
from app.main import main

@pytest.fixture
def config():
    return {
        'csv_file': 'test.csv',
        'headers': True,
        'window_size': 10,
        'max_error': 0.01,
        'initial_size': 256,
        'step_size': 32,
        'save_encoder': 'encoder',
        'output_file': 'output',
        'force_date': False,
        'remote_log': 'http://localhost:60500/preprocessor/feature_extractor/create',
        'remote_username': 'test',
        'remote_password': 'pass',
        'encoder_plugin': 'default_encoder'
    }

@pytest.fixture
def debug_info():
    return {
        "execution_time": "",
        "input_rows": 0,
        "output_rows": 0,
        "input_columns": 0,
        "output_columns": 0
    }

@patch("app.cli.parse_args")
@patch("app.config_handler.load_config")
@patch("app.config_handler.save_config")
@patch("app.config_handler.save_debug_info")
@patch("app.config_handler.merge_config")
@patch("app.data_handler.load_csv")
@patch("app.data_processor.process_data")
@patch("requests.post")
def test_main(mock_requests_post, mock_process_data, mock_load_csv, mock_merge_config, mock_save_debug_info, mock_save_config, mock_load_config, mock_parse_args, config, debug_info):
    # Mock parse_args
    mock_parse_args.return_value = MagicMock(), []

    # Mock load_config
    mock_load_config.return_value = config

    # Mock merge_config
    merged_config = config.copy()
    mock_merge_config.return_value = merged_config

    # Mock load_csv
    mock_load_csv.return_value = pd.DataFrame(np.random.rand(100, 10))

    # Mock process_data
    mock_process_data.return_value = (pd.DataFrame(np.random.rand(90, 10)), debug_info)

    # Mock save_config
    mock_save_config.return_value = (json.dumps(config), 'config_out.json')

    # Mock requests.post
    mock_requests_post.return_value.text = "Logged"

    # Mock sys.argv to control the command-line arguments
    with patch('sys.argv', ['main.py', 'test.csv']):
        # Capture the output
        with patch('sys.stdout', new_callable=lambda: MagicMock()) as mock_stdout:
            main()

    # Assertions
    mock_parse_args.assert_called_once()
    mock_load_config.assert_called_once()
    mock_merge_config.assert_called_once()
    mock_load_csv.assert_called_once_with(config['csv_file'], headers=config['headers'])
    mock_process_data.assert_called_once_with(config)
    mock_save_config.assert_called_once_with(config)
    mock_save_debug_info.assert_called_once_with(debug_info, 'debug_out.json')
    mock_requests_post.assert_called_once()

    # Verify output
    output = "".join([call[0][0] for call in mock_stdout.write.call_args_list])
    assert "Parsing initial arguments..." in output
    assert "Loading configuration..." in output
    assert "Merging configuration with CLI arguments..." in output
    assert "Processing complete. Writing output..." in output
    assert "Configuration saved to config_out.json" in output
    assert "Debug info saved to debug_out.json" in output
    assert "Execution time:" in output

@patch("app.cli.parse_args")
@patch("app.config_handler.load_config")
@patch("app.config_handler.merge_config")
def test_main_no_csv_file(mock_merge_config, mock_load_config, mock_parse_args, config):
    # Mock parse_args
    mock_parse_args.return_value = MagicMock(), []

    # Mock load_config
    mock_load_config.return_value = {}

    # Mock merge_config
    mock_merge_config.return_value = {}

    # Mock sys.argv to control the command-line arguments
    with patch('sys.argv', ['main.py']):
        # Capture the output
        with patch('sys.stderr', new_callable=lambda: MagicMock()) as mock_stderr:
            with pytest.raises(SystemExit):
                main()

        # Verify output
        output = "".join([call[0][0] for call in mock_stderr.write.call_args_list])
        assert "Error: No CSV file specified." in output

@patch("app.cli.parse_args")
@patch("app.config_handler.load_config")
@patch("app.config_handler.merge_config")
@patch("app.data_handler.load_csv")
def test_main_invalid_csv(mock_load_csv, mock_merge_config, mock_load_config, mock_parse_args, config):
    # Mock parse_args
    mock_parse_args.return_value = MagicMock(), []

    # Mock load_config
    mock_load_config.return_value = config

    # Mock merge_config
    merged_config = config.copy()
    merged_config['csv_file'] = 'invalid.csv'
    mock_merge_config.return_value = merged_config

    # Mock load_csv to raise FileNotFoundError
    mock_load_csv.side_effect = FileNotFoundError

    # Mock sys.argv to control the command-line arguments
    with patch('sys.argv', ['main.py', 'invalid.csv']):
        # Capture the output
        with patch('sys.stderr', new_callable=lambda: MagicMock()) as mock_stderr:
            with pytest.raises(SystemExit):
                main()

        # Verify output
        output = "".join([call[0][0] for call in mock_stderr.write.call_args_list])
        assert "Error: The file invalid.csv does not exist." in output
