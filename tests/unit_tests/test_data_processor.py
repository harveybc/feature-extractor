import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from app.data_processor import process_data
from app.data_handler import load_csv, write_csv, sliding_window
from app.plugin_loader import load_plugin

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
        'remote_password': 'pass'
    }

@patch("app.data_handler.load_csv")
@patch("app.data_handler.write_csv")
@patch("app.data_handler.sliding_window")
@patch("app.plugin_loader.load_plugin")
@patch("requests.post")
def test_process_data(mock_requests_post, mock_load_plugin, mock_sliding_window, mock_write_csv, mock_load_csv, config):
    # Mock load_csv
    mock_load_csv.return_value = pd.DataFrame(np.random.rand(100, 10))
    
    # Mock sliding_window
    mock_sliding_window.return_value = [np.random.rand(90, 10)]
    
    # Mock encoder
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = np.random.rand(90, 256)
    mock_encoder.calculate_mse.return_value = 0.005
    mock_load_plugin.return_value.return_value = mock_encoder
    
    # Mock requests.post
    mock_requests_post.return_value.text = "Logged"

    decoded_data, debug_info = process_data(config)
    
    assert 'mean_squared_error_0' in debug_info
    assert debug_info['mean_squared_error_0'] == 0.005
    mock_write_csv.assert_called_once()
    mock_requests_post.assert_called_once()
