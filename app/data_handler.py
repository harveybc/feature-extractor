import pandas as pd

def load_csv(file_path, headers=True):
    print(f"Loading CSV file from path: {file_path} with headers={headers}")
    data = pd.read_csv(file_path, header=0 if headers else None)
    print(f"Data loaded: {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

def write_csv(file_path, data, include_date=False, headers=True):
    print(f"Writing CSV file to path: {file_path} with headers={headers}")
    data.to_csv(file_path, index=False, header=headers)
    print("CSV file written successfully.")

def sliding_window(file_path, window_size, data):
    print(f"Applying sliding window of size {window_size} to data from {file_path}")
    windowed_data = []
    for i in range(len(data) - window_size + 1):
        windowed_data.append(data.iloc[i:i + window_size].values)
    print(f"Generated {len(windowed_data)} windows.")
    return windowed_data
