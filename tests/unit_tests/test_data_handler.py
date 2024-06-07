import pytest
import pandas as pd
import numpy as np
from app.data_handler import load_csv, sliding_window, write_csv

def test_load_csv_with_headers():
    with pytest.raises(FileNotFoundError):
        load_csv('non_existent_file.csv', headers=True)

def test_load_csv_without_headers():
    data = pd.DataFrame({
        'date': pd.date_range(start='1/1/2021', periods=5),
        'col_1': [1, 2, 3, 4, 5],
        'col_2': [5, 4, 3, 2, 1]
    })
    data.to_csv('test_no_headers.csv', index=False, header=False)
    
    loaded_data = load_csv('test_no_headers.csv', headers=False)
    pd.testing.assert_frame_equal(loaded_data.reset_index(drop=True), data)

def test_sliding_window():
    data = pd.DataFrame({
        'col_1': [1, 2, 3, 4, 5],
        'col_2': [5, 4, 3, 2, 1]
    })
    windows = sliding_window(data, window_size=3)
    expected_windows = [
        np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]),
        np.array([[5, 4, 3], [4, 3, 2], [3, 2, 1]])
    ]
    for win, expected in zip(windows, expected_windows):
        np.testing.assert_array_equal(win, expected)

def test_write_csv():
    data = pd.DataFrame({
        'date': pd.date_range(start='1/1/2021', periods=5),
        'col_1': [1, 2, 3, 4, 5],
        'col_2': [5, 4, 3, 2, 1]
    })
    write_csv('test_write.csv', data, include_date=True, headers=True)
    
    loaded_data = pd.read_csv('test_write.csv', parse_dates=['date'])
    pd.testing.assert_frame_equal(loaded_data, data)

    write_csv('test_write_no_headers.csv', data, include_date=True, headers=False)
    
    loaded_data_no_headers = pd.read_csv('test_write_no_headers.csv', header=None, parse_dates=[0])
    expected_data_no_headers = data.copy()
    expected_data_no_headers.columns = [0, 1, 2]
    pd.testing.assert_frame_equal(loaded_data_no_headers, expected_data_no_headers)
