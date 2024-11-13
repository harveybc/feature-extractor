import numpy as np
import pandas as pd
import argparse
import os

def generate_positional_encoding(num_positions, num_features):
    """
    Generates a positional encoding matrix of size (num_positions, num_features).
    
    Args:
        num_positions (int): Number of positions in the time series (total length of the combined dataset).
        num_features (int): Number of dimensions for the positional encoding, must be even.

    Returns:
        np.ndarray: Positional encoding matrix of size (num_positions, num_features).
    """
    encoding = np.zeros((num_positions, num_features))
    
    for pos in range(num_positions):
        for i in range(0, num_features, 2):
            div_term = 10000 ** (2 * (i // 2) / num_features)
            encoding[pos, i] = np.sin(pos / div_term)
            encoding[pos, i + 1] = np.cos(pos / div_term)
    
    return encoding

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Add positional encoding to multiple CSV files.")
    parser.add_argument("input_file", type=str, help="Filename of the first CSV input file.")
    parser.add_argument("--file2", type=str, default="file2.csv", help="Filename of the second CSV input file (default: file2.csv).")
    parser.add_argument("--file3", type=str, default="file3.csv", help="Filename of the third CSV input file (default: file3.csv).")
    parser.add_argument("--output1", type=str, default="pos_encoded_file1.csv", help="Output filename for the first file (default: pos_encoded_file1.csv).")
    parser.add_argument("--output2", type=str, default="pos_encoded_file2.csv", help="Output filename for the second file (default: pos_encoded_file2.csv).")
    parser.add_argument("--output3", type=str, default="pos_encoded_file3.csv", help="Output filename for the third file (default: pos_encoded_file3.csv).")
    
    args = parser.parse_args()
    
    # Load the datasets and check if all files exist
    file_list = [args.input_file, args.file2, args.file3]
    dfs = []
    for file in file_list:
        if not os.path.exists(file):
            print(f"Error: The file '{file}' does not exist.")
            return
        dfs.append(pd.read_csv(file, header=None))

    # Calculate total positions and features for positional encoding
    num_features = dfs[0].shape[1]  # Assume all files have the same number of columns
    num_positions = sum(len(df) for df in dfs)  # Total number of rows across all files
    
    # Generate positional encoding for the entire dataset sequence
    encoding = generate_positional_encoding(num_positions, num_features)
    
    # Split the encoding based on the length of each dataset and add to each DataFrame
    start_idx = 0
    for i, df in enumerate(dfs):
        end_idx = start_idx + len(df)
        df_encoding = encoding[start_idx:end_idx, :]
        
        # Add positional encoding columns to the DataFrame
        for j in range(num_features):
            df[f'pos_enc_{j}'] = df_encoding[:, j]
        
        # Save each DataFrame with positional encoding to a new CSV file
        output_filename = getattr(args, f"output{i+1}")
        df.to_csv(output_filename, index=False, header=False)
        print(f"Positional encoding added and saved to '{output_filename}'.")
        
        # Update the starting index for the next file
        start_idx = end_idx

if __name__ == "__main__":
    main()
