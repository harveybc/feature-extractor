import numpy as np

def reconstruct_series_from_windows(windowed_data, original_length, window_size):
    """
    Reconstruct the original time series from windowed data by averaging the overlapping windows.
    
    Args:
        windowed_data (np.array): The windowed data, shape (num_windows, window_size).
        original_length (int): The original length of the time series.
        window_size (int): The size of each window.
    
    Returns:
        np.array: The reconstructed time series of the original length.
    """
    num_windows = len(windowed_data)
    overlap = window_size - 1
    
    reconstructed_series = np.zeros(original_length)
    window_counts = np.zeros(original_length)
    
    for i in range(num_windows):
        start_idx = i * overlap
        end_idx = start_idx + window_size
        reconstructed_series[start_idx:end_idx] += windowed_data[i, :]
        window_counts[start_idx:end_idx] += 1
    
    # Avoid division by zero
    window_counts[window_counts == 0] = 1
    
    reconstructed_series /= window_counts
    
    return reconstructed_series

# Example usage
windowed_data = ... # Your windowed data (num_windows, window_size)
original_length = 73841  # Length of the original time series
window_size = 512  # Size of each window

reconstructed_series = reconstruct_series_from_windows(windowed_data, original_length, window_size)
print(f"Reconstructed series shape: {reconstructed_series.shape}")
print(f"First 5 rows of reconstructed series: {reconstructed_series[:5]}")
