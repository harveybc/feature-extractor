import pandas as pd
import numpy as np

def sliding_window(file_path, window_size, data):
    """
    Load a CSV file and process it using a sliding window technique.

    Args:
        file_path (str): The path to the CSV file to be loaded.
        window_size (int): The size of the sliding window to apply.

    Returns:
        list of np.array: A list of arrays, each containing sliding window data for one column.
    """
    try:
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

def load_csv(file_path, headers=False):
    """
    Load a CSV file into a pandas DataFrame, handling date columns and correct numeric parsing.

    Args:
        file_path (str): The path to the CSV file to be loaded.
        headers (bool): Whether the CSV file has headers.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        if headers:
            data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
        else:
            # Read the file without headers
            data = pd.read_csv(file_path, header=None, sep=',', parse_dates=False)
            first_col = data.iloc[:, 0]

            # Attempt to parse the first column as dates
            try:
                data[0] = pd.to_datetime(first_col, dayfirst=True)
                if data[0].notna().all():
                    data.columns = ['date']
                    data.set_index('date', inplace=True)
            except (ValueError, pd.errors.ParserError):
                # If parsing fails, treat the column as numeric
                data.columns = [f'col_{i}' for i in range(len(data.columns))]
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

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
        print(f"An error occurred while loading the CSV: {e}")
        raise

    return data

def write_csv(file_path, data, include_date=True, headers=True):
    """
    Write a pandas DataFrame to a CSV file.

    Args:
        file_path (str): The path to the CSV file to be written.
        data (pd.DataFrame): The data to be written to the CSV file.
        include_date (bool): Whether to include the date column in the output.
        headers (bool): Whether to include headers in the output.

    Returns:
        None
    """
    try:
        if include_date and 'date' in data.columns:
            data.to_csv(file_path, index=True, header=headers)
        else:
            data.to_csv(file_path, index=False, header=headers)
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")
        raise
