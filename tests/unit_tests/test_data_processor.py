import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
from app.data_processor import process_data

@patch("builtins.open", new_callable=mock_open, read_data="2024-01-01,1,2,3\n2024-01-02,4,5,6\n2024-01-03,7,8,9")
@patch("app.data_handler.load_csv")
@patch("app.data_handler.write_csv")
@patch("app.data_handler.sliding_window")
@patch("app.plugin_loader.load_encoder_decoder_plugins")
@patch("requests.post")
def test_process_data(mock_requests_post, mock_load_encoder_decoder_plugins, mock_sliding_window, mock_write_csv, mock_load_csv, mock_file):
    mock_load_csv.return_value = pd.read_csv('tests/data/csv_sel_unb_norm_512.csv')
    mock_sliding_window.return_value
