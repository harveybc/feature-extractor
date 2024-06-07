import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.main import main

@pytest.fixture
def mock_parse_args():
    return MagicMock(
        load_config='tests/data/config.json',
        remote_load_config=None,
        save_config='tests/data/config_out.json',
        debug_file='debug_out.json',
        csv_file='tests/data/csv_sel_unb_norm_512.csv',
        remote_load_encoder=None,
        remote_load_decoder=None
    ), []

def test_load_and_merge_config(mock_parse_args):
    with patch("app.cli.parse_args", return_value=mock_parse_args):
        with patch("app.config_handler.load_config", return_value={'max_error': 0.05, 'initial_size': 128}):
            with patch("app.config_handler.save_config") as mock_save_config:
                with patch("app.config_handler.save_debug_info"):
                    with patch("app.config_handler.merge_config", side_effect=lambda config, cli_args, plugin_params: {**config, **cli_args, **plugin_params}):
                        with patch("app.data_handler.load_csv", return_value=pd.read_csv('tests/data/csv_sel_unb_norm_512.csv')):
                            with patch("app.data_processor.process_data", return_value=(pd.DataFrame(np.random.rand(10, 10)), {'execution_time': 0})):
                                with patch("requests.post"):
                                    main()
                                    expected_config = {
                                        'csv_file': 'tests/data/csv_sel_unb_norm_512.csv',
                                        'headers': True,
                                        'window_size': 10,
                                        'max_error': 0.05,
                                        'initial_size': 128,
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
                                    mock_save_config.assert_called_once_with(expected_config, 'tests/data/config_out.json')

def test_save_final_config(mock_parse_args):
    with patch("app.cli.parse_args", return_value=mock_parse_args):
        with patch("app.config_handler.load_config", return_value={'max_error': 0.05, 'initial_size': 128}):
            with patch("app.config_handler.save_config") as mock_save_config:
                with patch("app.config_handler.save_debug_info"):
                    with patch("app.config_handler.merge_config", side_effect=lambda config, cli_args, plugin_params: {**config, **cli_args, **plugin_params}):
                        with patch("app.data_handler.load_csv", return_value=pd.read_csv('tests/data/csv_sel_unb_norm_512.csv')):
                            with patch("app.data_processor.process_data", return_value=(pd.DataFrame(np.random.rand(10, 10)), {'execution_time': 0})):
                                with patch("requests.post"):
                                    main()
                                    mock_save_config.assert_called_once_with({
                                        'csv_file': 'tests/data/csv_sel_unb_norm_512.csv',
                                        'headers': True,
                                        'window_size': 10,
                                        'max_error': 0.05,
                                        'initial_size': 128,
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
                                    }, 'tests/data/config_out.json')

def test_save_debug_info(mock_parse_args):
    with patch("app.cli.parse_args", return_value=mock_parse_args):
        with patch("app.config_handler.load_config", return_value={'max_error': 0.05, 'initial_size': 128}):
            with patch("app.config_handler.save_config"):
                with patch("app.config_handler.save_debug_info") as mock_save_debug_info:
                    with patch("app.config_handler.merge_config", side_effect=lambda config, cli_args, plugin_params: {**config, **cli_args, **plugin_params}):
                        with patch("app.data_handler.load_csv", return_value=pd.read_csv('tests/data/csv_sel_unb_norm_512.csv')):
                            with patch("app.data_processor.process_data", return_value=(pd.DataFrame(np.random.rand(10, 10)), {'execution_time': 0})):
                                with patch("requests.post"):
                                    main()
                                    mock_save_debug_info.assert_called_once_with({
                                        'execution_time': 0,
                                        'input_rows': 512,  # Assuming the mock CSV has 512 rows
                                        'input_columns': 5,  # Assuming the mock CSV has 5 columns
                                        'output_rows': 10,
                                        'output_columns': 10
                                    }, 'debug_out.json')

def test_remote_config_load_and_save(mock_parse_args):
    with patch("app.cli.parse_args", return_value=mock_parse_args):
        with patch("app.config_handler.load_config", return_value={}):
            with patch("app.config_handler.save_config"):
                with patch("app.config_handler.save_debug_info"):
                    with patch("app.config_handler.merge_config", side_effect=lambda config, cli_args, plugin_params: {**config, **cli_args, **plugin_params}):
                        with patch("app.config_handler.load_remote_config", return_value={'max_error': 0.05, 'initial_size': 128}) as mock_load_remote_config:
                            with patch("app.config_handler.save_remote_config") as mock_save_remote_config:
                                with patch("app.data_handler.load_csv", return_value=pd.read_csv('tests/data/csv_sel_unb_norm_512.csv')):
                                    with patch("app.data_processor.process_data", return_value=(pd.DataFrame(np.random.rand(10, 10)), {'execution_time': 0})):
                                        with patch("requests.post"):
                                            main()
                                            mock_load_remote_config.assert_called_once_with(
                                                'http://localhost:60500/preprocessor/feature_extractor/detail/1',
                                                'test',
                                                'pass'
                                            )
                                            mock_save_remote_config.assert_called_once()
