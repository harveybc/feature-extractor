import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from app.data_handler import load_csv, write_csv

@patch("builtins.open", new_callable=mock_open, read_data="2024-01-01,1,2,3\n2024-01-02,4,5,6\n2024-01-03,7,8,9")
def test_load_csv_without_headers(mock_file):
    df = load_csv("test.csv", headers=False)
    expected_df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'col_0': [1, 2, 3],
        'col_1': [4, 5, 6],
        'col_2': [7, 8, 9]
    })
    pd.testing.assert_frame_equal(df.reset_index(), expected_df)

@patch("builtins.open", new_callable=mock_open)
@patch("pandas.DataFrame.to_csv")
def test_write_csv(mock_to_csv, mock_open):
    data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=3, freq='D'),
        'col_0': [1, 4, 7],
        'col_1': [2, 5, 8],
        'col_2': [3, 6, 9]
    })
    write_csv("test_output.csv", data)
    mock_to_csv.assert_called_once_with("test_output.csv", index=True, header=True)
