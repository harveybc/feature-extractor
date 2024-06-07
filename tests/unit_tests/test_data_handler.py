import pytest
import pandas as pd
from app.data_handler import load_csv, write_csv

def test_load_csv_without_headers():
    data = "1,2,3\n4,5,6\n7,8,9"
    with patch("builtins.open", mock_open(read_data=data)):
        df = load_csv("test.csv", headers=False)
    expected_df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pd.testing.assert_frame_equal(df, expected_df)

@patch("pandas.DataFrame.to_csv")
def test_write_csv(mock_to_csv):
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    write_csv("test.csv", df, include_date=False, headers=False)
    mock_to_csv.assert_called_once_with("test.csv", index=False, header=False)
