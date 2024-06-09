import numpy as np
import pandas as pd

def load_csv(file_path, headers=False):
    print(f"Loading CSV file from path: {file_path} with headers={headers}")
    data = pd.read_csv(file_path, header=0 if headers else None)
    print(f"Data loaded: {len(data)} rows and {data.shape[1]} columns.")
    return data

def write_csv(file_path, data, include_date=False, headers=False):
    print(f"Writing data to CSV file at path: {file_path} with headers={headers} and include_date={include_date}")
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if include_date:
        data.insert(0, 'Date', pd.date_range(start='1/1/2000', periods=len(data), freq='D'))
    data.to_csv(file_path, index=False, header=headers)
    print(f"Data written to {file_path}")

def sliding_window(data, window_size):
    print(f"Applying sliding window of size: {window_size}")
    data = np.array(data)  # Ensure data is a numpy array
    print(f"Data shape: {data.shape}")
    
    if window_size > data.shape[0]:
        print(f"Warning: Window size {window_size} is larger than the data length {data.shape[0]}.")
        window_size = data.shape[0]
        
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    
    print(f"Shape for as_strided: {shape}")
    print(f"Strides for as_strided: {strides}")
    
    windowed_data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    
    print(f"Windowed data shape: {windowed_data.shape}")
    return windowed_data
