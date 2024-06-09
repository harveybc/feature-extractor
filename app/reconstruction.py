import numpy as np

def reconstruct_from_windows(windowed_data, original_length, window_size):
    """
    Reconstruct the original timeseries from the windowed data by averaging the overlapping windows.

    Args:
        windowed_data (np.array): The windowed data from the autoencoder.
        original_length (int): The length of the original timeseries.
        window_size (int): The size of the window used for sliding.

    Returns:
        np.array: The reconstructed original timeseries.
    """
    reconstructed_data = np.zeros((original_length, 1))
    counts = np.zeros((original_length, 1))

    for i in range(windowed_data.shape[0]):
        for j in range(window_size):
            if i + j < original_length:
                reconstructed_data[i + j] += windowed_data[i, j]
                counts[i + j] += 1

    reconstructed_data = reconstructed_data / counts
    return reconstructed_data

# Debug function to test reconstruction
def test_reconstruct():
    windowed_data = np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7]
    ])
    original_length = 9
    window_size = 5
    reconstructed = reconstruct_from_windows(windowed_data, original_length, window_size)
    print(f"Reconstructed data: {reconstructed.flatten()}")

# Uncomment to run the test
# test_reconstruct()
