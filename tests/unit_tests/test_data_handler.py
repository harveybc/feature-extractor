import pytest
import pandas as pd
import numpy as np
import os
from app.data_handler import load_csv, write_csv, sliding_window

# Setup: Create a sample DataFrame for testing
@pytest.fixture
def sample_dataframe():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1]
    }
    return pd.DataFrame(data)

def test_load_csv(tmp_path, sample_dataframe):
    # Test Case 1: Load a CSV file
    test_csv_path = tmp_path / "test_load.csv"
    sample_dataframe.to_csv(test_csv_path, index=False)
    loaded_df = load_csv(test_csv_path, headers=True)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    # Test Case 2: Handle missing values
    sample_dataframe.loc[2, 'A'] = np.nan
    sample_dataframe.to_csv(test_csv_path, index=False)
    loaded_df = load_csv(test_csv_path, headers=True)
    assert loaded_df.isna().sum().sum() == 1  # Check for one missing value
    
    # Test Case 3: Non-existent file
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file.csv", headers=True)

def test_write_csv(tmp_path, sample_dataframe):
    # Test Case 1: Write DataFrame to CSV and verify content
    test_csv_path = tmp_path / "test_write.csv"
    write_csv(test_csv_path, sample_dataframe.values, include_date=False, headers=sample_dataframe.columns)
    loaded_df = pd.read_csv(test_csv_path)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    # Test Case 2: Handle permission denied
    with pytest.raises(PermissionError):
        # Simulate permission error by writing to a restricted directory
        restricted_path = "Z:/restricted_test_write.csv"  # Update with an appropriate path
        write_csv(restricted_path, sample_dataframe.values, include_date=False, headers=sample_dataframe.columns)

def test_sliding_window():
    # Test Case 1: Simple sliding window
    arr = np.array([1, 2, 3, 4, 5])
    window_size = 3
    result = sliding_window(arr, window_size)
    expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    np.testing.assert_array_equal(result, expected)
    
    # Test Case 2: Window size greater than array length
    window_size = 6
    result = sliding_window(arr, window_size)
    assert result.size == 0  # Expect an empty array
    
    # Test Case 3: Overlapping windows
    window_size = 2
    result = sliding_window(arr, window_size)
    expected = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    np.testing.assert_array_equal(result, expected)

if __name__ == "__main__":
    pytest.main()
