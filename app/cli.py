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
    parser.add_argument('-em', '--encoder_plugin', type=str, help='Name of the encoder plugin to use.')
    parser.add_argument('-dm', '--decoder_plugin', type=str, help='Name of the decoder plugin to use.')

    # Optional argument for specifying minimum MSE to stop training and export the model
    parser.add_argument('-me', '--min_mse', type=float, help='Minimum MSE error to stop the training process and export the model.')

    return parser.parse_args()

if __name__ == '__main__':
    # Parse the command line arguments
    args = parse_args()

    # Example: print the parsed arguments (this line can be removed or replaced with actual logic)
    print(args)
