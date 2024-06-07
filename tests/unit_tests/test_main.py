import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.main import main
import json

@patch("app.cli.parse_args")
@patch("app.config_handler.load_config")
@patch("app.config_handler.save_config")
@patch("app.config_handler.save_debug_info")
@patch("app.config_handler.merge_config")
@patch("app.data_handler.load_csv")
@patch("app.data_processor.process_data")
@patch("requests.post")
def test_main(mock_requests_post, mock_process_data, mock_load_csv, mock_merge_config, mock_save_debug_info, mock_save_config, mock_load_config, mock_parse_args):
    # Mock parse_args
    mock_parse_args.return_value = MagicMock(load_config=None, remote_load_config=None, save_config=None, debug_file=None, csv_file='tests/data/csv_sel_unb_norm_512.csv', remote_load_encoder=None, remote_load_decoder=None), []

    # Mock load_config
    mock_load_config.return_value = {}

    # Mock merge_config
    mock_merge_config.return_value = {
        'csv_file': 'tests/data/csv_sel_unb_norm_512.csv',
        'headers': True,
        'window_size': 10,
        'max_error': 0.01,
        'initial_size': 256,
        'step_size': 32,
        'save_encoder': 'encoder_model.h5',
        'save_decoder': 'decoder_model.h5',
        'output_file': 'output.csv',
        'force_date': False,
        'remote_log': 'http://localhost:60500/preprocessor/feature_extractor/create',
        'remote_username': 'test',
        'remote_password': 'pass',
        'encoder_plugin': 'default_encoder',
        'decoder_plugin': 'default_decoder'
    }

    # Mock load_csv
    mock_load_csv.return_value = pd.read_csv('tests/data/csv_sel_unb_norm_512.csv')

    # Mock process_data
    mock_process_data.return_value = (pd.DataFrame(np.random.rand(10, 10)), {'execution_time': 0})

    # Mock save_config
    mock_save_config.return_value = (json.dumps(mock_merge_config.return_value), 'config_out.json')

    # Mock requests.post
    mock_requests_post.return_value.text = "Logged"

    # Capture the output
    with patch('sys.stdout', new_callable=lambda: MagicMock()) as mock_stdout:
        main()
