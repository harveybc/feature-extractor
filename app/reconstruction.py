import numpy as np
import pandas as pd

def unwindow_data(windowed_df):
    """
    Transform a windowed dataset into a non-windowed dataset for each column independently.
    
    Parameters:
    windowed_df (DataFrame): The input dataset with windowed data where each column is a channel.
    
    Returns:
    DataFrame: The resulting non-windowed dataset with all channels reconstructed.
    """
    window_size = windowed_df.shape[1]  # Total number of values across all channels
    num_rows = len(windowed_df)
    
    # Calculate the total rows for the output, assuming the same logic as before
    total_rows_out = num_rows + window_size - 1

    # Create a DataFrame to hold the unwindowed data for each channel (each column)
    output_dataset = pd.DataFrame(0, index=range(total_rows_out), columns=windowed_df.columns)
    print(f"[unwindow_data] Un-windowing output data for {len(windowed_df.columns)} columns")

    percen_val = num_rows // 100
    count = 0
    
    # Process each column (each channel) independently
    for col in windowed_df.columns:
        print(f"[unwindow_data] Processing column: {col}")
        for row in range(num_rows):
            if count == percen_val:
                print(f"{row // percen_val}% done", end="\r", flush=True)
                count = 0
            count += 1
            
            # Create an extended row (sliding window) and add it to the output dataset
            extended_row = np.zeros(total_rows_out)
            extended_row[row:row + window_size] = windowed_df.iloc[row][col]
            output_dataset[col] += extended_row

        # Calculate the averages as done in the original version
        print(f"[unwindow_data] Calculating averages in the first segment for {col}")
        for row in range(window_size - 1):
            output_dataset.iloc[row][col] /= (row + 1)
        
        print(f"[unwindow_data] Calculating averages in the second segment for {col}")
        for row in range(window_size - 1, total_rows_out - window_size):
            output_dataset.iloc[row][col] /= window_size
        
        print(f"[unwindow_data] Calculating averages in the last segment for {col}")
        for row in range(total_rows_out - window_size, total_rows_out):
            output_dataset.iloc[row][col] /= (total_rows_out - row)

    print("[unwindow_data] Unwindowing completed for all columns.")
    
    return output_dataset
