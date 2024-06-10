import numpy as np
import pandas as pd

def unwindow_data(windowed_df):
    """
    Transform a windowed dataset into a non-windowed dataset by following a precise procedure.
    
    Parameters:
    windowed_df (DataFrame): The input dataset with windowed data.
    
    Returns:
    DataFrame: The resulting non-windowed dataset.
    """
    window_size = windowed_df.shape[1]
    num_rows = len(windowed_df)
    total_rows_out = num_rows + window_size

    output_dataset = pd.DataFrame(0, index=range(total_rows_out), columns=['Output'])

    for row in range(num_rows):
        extended_row = np.zeros(total_rows_out)
        extended_row[row:row + window_size] = windowed_df.iloc[row].values
        output_dataset['Output'] += extended_row

    for row in range(window_size - 2):
        output_dataset.iloc[row] /= (row + 1)
    for row in range(window_size - 1, total_rows_out - window_size):
        output_dataset.iloc[row] /= window_size
    for row in range(total_rows_out - window_size+1, total_rows_out-1):
        output_dataset.iloc[row] /= (total_rows_out - row)

    return output_dataset
