import pandas as pd
from app.reconstruction import unwindow_data

# app/data_handler.py

def load_csv(file_path, headers=False, force_date=False):
    try:
        if headers:
            if force_date:
                # Assume the first column is 'date'
                data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
                data.set_index(data.columns[0], inplace=True)
                data.index.name = 'date'
                # Convert all other columns to numeric, coercing errors to NaN
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            else:
                # Headers present but no date column
                data = pd.read_csv(file_path, sep=',', dayfirst=True)
                # Convert all columns to numeric, coercing errors to NaN
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
        else:
            if force_date:
                # No headers but the first column is 'date'
                data = pd.read_csv(file_path, header=None, sep=',', parse_dates=[0], dayfirst=True)
                data.columns = ['date'] + [f'col_{i}' for i in range(1, len(data.columns))]
                data.set_index('date', inplace=True)
                # Convert all other columns to numeric, coercing errors to NaN
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            else:
                # No headers and no date column
                data = pd.read_csv(file_path, header=None, sep=',', dayfirst=True)
                data.columns = [f'col_{i}' for i in range(len(data.columns))]
                # Convert all columns to numeric, coercing errors to NaN
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        raise
    return data



# app/data_handler.py

def write_csv(file_path, data, include_date=True, headers=True, force_date=False):
    try:
        if include_date and force_date and 'date' in data.index.names:
            data.to_csv(file_path, index=True, header=headers)
        else:
            data.to_csv(file_path, index=False, header=headers)
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")
        raise
