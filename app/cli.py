import argparse

def parse_args():
    """
    Parses command line arguments provided to the feature-extractor application.

    Returns:
        argparse.Namespace: The namespace containing the arguments and their values.
    """
    parser = argparse.ArgumentParser(description="Feature-extractor: A tool for encoding and decoding CSV data with support for dynamic plugins.")

    # Required positional argument for the input CSV file
    parser.add_argument('csv_file', type=str, help='Path to the CSV file to process.')

    # Optional arguments for saving and loading model parameters
    parser.add_argument('-ds', '--save_encoder', type=str, help='Filename to save the trained encoder model.')
    parser.add_argument('-dl', '--load_decoder_params', type=str, help='Filename to load decoder parameters from.')
    parser.add_argument('-el', '--load_encoder_params', type=str, help='Filename to load encoder parameters from.')

    # Optional arguments for evaluating encoder and decoder
    parser.add_argument('-ee', '--evaluate_encoder', type=str, help='Filename for outputting encoder evaluation results.')
    parser.add_argument('-de', '--evaluate_decoder', type=str, help='Filename for outputting decoder evaluation results.')

    # Optional arguments for selecting plugins
    parser.add_argument('-ep', '--encoder_plugin', type=str, default='default_encoder', help='Name of the encoder plugin to use.')
    parser.add_argument('-dp', '--decoder_plugin', type=str, default='default_decoder', help='Name of the decoder plugin to use.')


    # Optional argument for specifying the sliding window size
    parser.add_argument('-ws', '--window_size', type=int, default=10, help='Sliding window size to use for processing time series data.')
    
    # Arguments related to autoencoder configuration and training
    parser.add_argument('-me', '--max_error', type=float, help='Maximum MSE error to stop the training process.')
    parser.add_argument('-is', '--initial_size', type=int, help='Initial size of the output of the encoder/input of the decoder.')
    parser.add_argument('-ss', '--step_size', type=int, help='Step size to reduce the size of the encoder/decoder interface on each iteration.')

    return parser.parse_args()

if __name__ == '__main__':
    # Parse the command line arguments
    args = parse_args()

    # Example: print the parsed arguments (this line can be removed or replaced with actual logic)
    print(args)
