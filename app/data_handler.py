import pandas as pd
import numpy as np

def load_csv(file_path, headers=True):
    print(f"Loading CSV file from path: {file_path} with headers={headers}")
    if headers:
        data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
    else:
        data = pd.read_csv(file_path, sep=',', header=None, parse_dates=[0], dayfirst=True)
    return data

def write_csv(file_path, data, include_date=False, headers=True):
    print(f"Writing data to CSV file at path: {file_path} with headers={headers} and include_date={include_date}")
    if include_date:
        data.to_csv(file_path, index=False, header=headers)
    else:
        data.iloc[:, 1:].to_csv(file_path, index=False, header=headers)

def sliding_window(data, window_size):
    print(f"Creating sliding windows of size: {window_size}")
    num_windows = len(data) - window_size + 1
    windows = [data[i:i + window_size].values for i in range(num_windows)]
    return np.array(windows)
