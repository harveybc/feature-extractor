import pandas as pd
import numpy as np

def load_csv(file_path, window_size):
    """
    Load a CSV file and process it using a sliding window technique.

    Args:
        file_path (str): The path to the CSV file to be loaded.
        window_size (int): The size of the sliding window to apply.

    Returns:
        list of np.array: A list of arrays, each containing sliding window data for one column.
    """
    try:
        data = pd.read_csv(file_path)
        series_list = []
        for column in data.columns:
            # Extract the column as array
            column_data = data[column].values
            # Apply sliding window
            windows = np.array([column_data[i:(i + window_size)] for i in range(len(column_data) - window_size + 1)])
            series_list.append(windows)
        return series_list
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        raise
    except pd.errors.ParserError:
        print("Error: Error parsing the file.")
        raise
    except Exception as e:
        print(f"An error occurred while loading and processing the CSV: {e}")
        raise
