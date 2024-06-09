import numpy as np

def sliding_window(data, window_size):
    print(f"Applying sliding window of size: {window_size}")
    data = np.array(data)  # Ensure data is a numpy array
    print(f"Data shape: {data.shape}")

    # Check if window size is greater than data length
    if window_size > data.shape[0]:
        print(f"Warning: Window size {window_size} is larger than the data length {data.shape[0]}.")
        return np.empty((0, window_size))

    # Ensure data is 2D for consistent indexing
    if data.ndim == 1:
        data = data[:, np.newaxis]

    # Calculate the correct shape and strides
    shape = (data.shape[0] - window_size + 1, window_size) + data.shape[1:]
    strides = (data.strides[0],) + data.strides
    windowed_data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return windowed_data
