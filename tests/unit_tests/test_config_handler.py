import pytest
from unittest.mock import patch, mock_open
import json
from app.config_handler import (
    load_config, save_config, merge_config, save_debug_info,
    load_remote_config, save_remote_config, log_remote_data
)

def test_load_config_success():
    with patch("builtins.open", mock_open(read_data='{"encoder_plugin": "test_encoder", "decoder_plugin": "test_decoder"}')) as mock_file:
        config = load_config('config.json')
        mock_file.assert_called_once_with('config.json', 'r')
        assert config == {"encoder_plugin": "test_encoder", "decoder_plugin": "test_decoder"}

def test_save_config_success():
    with patch("builtins.open", mock_open()) as mock_file:
        config = {"encoder_plugin": "test_encoder", "decoder_plugin": "test_decoder"}
        save_config(config, 'config.json')
        mock_file.assert_called_once_with('config.json', 'w')
        mock_file().write.assert_called_once_with(json.dumps(config, indent=4))

def test_merge_config():
    base_config = {"encoder_plugin": "default_encoder"}
    cli_args = {"encoder_plugin": "cli_encoder", "new_param": "value"}
    merged_config = merge_config(base_config, cli_args)
    assert merged_config == {"encoder_plugin": "cli_encoder", "new_param": "value"}

def test_save_debug_info():
    with patch("builtins.open", mock_open()) as mock_file:
        debug_info = {"step": "test"}
        save_debug_info(debug_info, 'debug.json')
        mock_file.assert_called_once_with('debug.json', 'w')
        mock_file().write.assert_called_once_with(json.dumps(debug_info, indent=4))

@patch("requests.get")
def test_load_remote_config(mock_get):
    mock_response = mock_get.return_value
    mock_response.json.return_value = {"encoder_plugin": "remote_encoder"}
    mock_response.raise_for_status = lambda: None
    config = load_remote_config()
    mock_get.assert_called_once_with('http://localhost:60500/preprocessor/feature_extractor/detail/1', auth=('test', 'pass'))
    assert config == {"encoder_plugin": "remote_encoder"}

@patch("requests.post")
def test_save_remote_config(mock_post):
    mock_response = mock_post.return_value
    mock_response.json.return_value = {"status": "success"}
    mock_response.raise_for_status = lambda: None
    response = save_remote_config({"encoder_plugin": "test_encoder"})
    mock_post.assert_called_once_with('http://localhost:60500/preprocessor/feature_extractor/create', auth=('test', 'pass'), json={"encoder_plugin": "test_encoder"})
    assert response == {"status": "success"}

@patch("requests.post")
def test_log_remote_data(mock_post):
    mock_response = mock_post.return_value
    mock_response.json.return_value = {"status": "logged"}
    mock_response.raise_for_status = lambda: None
    response = log_remote_data({"mse": 0.01})
    mock_post.assert_called_once_with('http://localhost:60500/preprocessor/feature_extractor/create', auth=('test', 'pass'), json={"mse": 0.01})
    assert response == {"status": "logged"}

