import pytest
import json
import requests
from unittest.mock import patch, mock_open
from app.config_handler import load_config, save_config, save_debug_info, merge_config, load_remote_config, save_remote_config, log_remote_data

def test_load_config():
    mock_file_data = '{"encoder_plugin": "default_encoder", "max_error": 0.01}'
    with patch("builtins.open", mock_open(read_data=mock_file_data)):
        config = load_config("mock_path")
        assert config == json.loads(mock_file_data)

def test_save_config():
    config = {'encoder_plugin': 'default_encoder', 'max_error': 0.01}
    with patch("builtins.open", mock_open()) as mock_file:
        config_str, path = save_config(config, "mock_path")
        mock_file().write.assert_called_once_with(json.dumps(config, indent=4))
        assert path == "mock_path"

def test_save_debug_info():
    debug_info = {"execution_time": 0.123, "input_rows": 100}
    with patch("builtins.open", mock_open()) as mock_file:
        save_debug_info(debug_info, "mock_debug_path")
        mock_file().write.assert_called_once_with(json.dumps(debug_info, indent=4))

def test_merge_config():
    default_config = {'encoder_plugin': 'default_encoder', 'max_error': 0.01}
    cli_args = {'encoder_plugin': 'custom_encoder', 'additional_param': 'value'}
    plugin_params = {'input_dim': 128, 'epochs': 100}
    merged_config = merge_config(default_config, cli_args, plugin_params)
    assert merged_config['encoder_plugin'] == 'custom_encoder'
    assert merged_config['max_error'] == 0.01
    assert merged_config['additional_param'] == 'value'
    assert merged_config['input_dim'] == 128
    assert merged_config['epochs'] == 100

@patch("requests.get")
def test_load_remote_config(mock_get):
    mock_get.return_value.json.return_value = {'encoder_plugin': 'remote_encoder', 'max_error': 0.02}
    mock_get.return_value.raise_for_status = lambda: None
    config = load_remote_config("mock_url", "user", "pass")
    assert config == {'encoder_plugin': 'remote_encoder', 'max_error': 0.02}

@patch("requests.post")
def test_save_remote_config(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status = lambda: None
    result = save_remote_config({'encoder_plugin': 'remote_encoder'}, "mock_url", "user", "pass")
    assert result == True

@patch("requests.post")
def test_log_remote_data(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status = lambda: None
    result = log_remote_data({'execution_time': 0.123}, "mock_url", "user", "pass")
    assert result == True
