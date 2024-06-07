import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
from app.data_processor import process_data

@patch("builtins.open", new_callable=mock_open, read_data="2024-01-01,1,2,3\n2024-01-02,4,5,6\n2024-01-03,7,8,9")
@patch("app.data_handler.load_csv")
@patch("app.data_handler.write_csv")
@patch("app.data_handler.sliding_window")
@patch("app.plugin_loader.load_plugin")
@patch("requests.post")
def test_process_data(mock_requests_post, mock_load_plugin, mock_sliding_window, mock_write_csv, mock_load_csv, mock_file):
    mock_load_csv.return_value = pd.read_csv('tests/data/csv_sel_unb_norm_512.csv')
    mock_sliding_window.return_value = [np.random.rand(90, 10)]
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = np.random.rand(90, 256)
    mock_encoder.model.predict.return_value = np.random.rand(90, 10)
    mock_encoder.calculate_mse.return_value = 0.005
    mock_load_plugin.return_value = mock_encoder
    mock_requests_post.return_value.text = "Logged"

    config = {
        'csv_file': 'tests/data/csv_sel_unb_norm_512.csv',
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

    decoded_data, debug_info = process_data(config)
    mock_load_csv.assert_called_once_with('tests/data/csv_sel_unb_norm_512.csv', headers=True)
    mock_write_csv.assert_called_once()
    pd.testing.assert_frame_equal(pd.DataFrame(decoded_data), pd.DataFrame(mock_encoder.model.predict.return_value))

if __name__ == "__main__":
    pytest.main()
