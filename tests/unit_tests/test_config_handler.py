import pytest
from unittest.mock import patch, mock_open, MagicMock
import json
from app.config_handler import save_config, load_remote_config, save_remote_config, log_remote_data, merge_config, save_debug_info

# Add requests import
import requests

@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_save_config_success(mock_json_dump, mock_open):
    config = {'key': 'value'}
    save_config(config, "config_out.json")
    mock_open.assert_called_once_with("config_out.json", 'w')
    mock_json_dump.assert_called_once_with(config, mock_open(), indent=4)

def test_merge_config():
    default_config = {'encoder_plugin': 'default_encoder', 'max_error': 0.01}
    cli_args = {'encoder_plugin': 'custom_encoder', 'additional_param': 'value'}
    merged_config = merge_config(default_config, cli_args)
    expected_config = {'encoder_plugin': 'custom_encoder', 'max_error': 0.01, 'additional_param': 'value'}
    assert merged_config == expected_config

@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_save_debug_info(mock_json_dump, mock_open):
    debug_info = {'execution_time': 123}
    save_debug_info(debug_info, "debug_out.json")
    mock_open.assert_called_once_with("debug_out.json", 'w')
    mock_json_dump.assert_called_once_with(debug_info, mock_open(), indent=4)

@patch("requests.get")
def test_load_remote_config(mock_requests_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {'config_key': 'config_value'}
    mock_requests_get.return_value = mock_response
    config = load_remote_config("http://localhost:60500/preprocessor/feature_extractor/detail/1", "test", "pass")
    assert config == {'config_key': 'config_value'}

@patch("requests.post")
def test_save_remote_config(mock_requests_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests_post.return_value = mock_response
    result = save_remote_config("config_string", "http://localhost:60500/preprocessor/feature_extractor/create", "test", "pass")
    assert result is True

@patch("requests.post")
def test_log_remote_data(mock_requests_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests_post.return_value = mock_response
    debug_info = {'execution_time': 123}
    result = log_remote_data(debug_info, "http://localhost:60500/preprocessor/feature_extractor/create", "test", "pass")
    assert result is True
