import pandas as pd

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
            data = pd.read_csv(file_path, header=None, sep=',')
            # Check if the first column is a date column
            if pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]) or pd.to_numeric(data.iloc[:, 0], errors='coerce').isna().all():
                data.columns = ['date'] + [f'col_{i-1}' for i in range(1, len(data.columns))]
                data.set_index('date', inplace=True)
            else:
                # Manually set column names if the first column is not a date
                data.columns = [f'col_{i}' for i in range(len(data.columns))]

            # Convert numeric columns to appropriate data types
            for col in data.columns:
                if col != 'date':
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
