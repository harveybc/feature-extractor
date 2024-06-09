import argparse
import config

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
    parser.add_argument('-se', '--save_encoder', type=str, default=config.SAVE_ENCODER_PATH, help='Filename to save the trained encoder model.')
    parser.add_argument('-sd', '--save_decoder', type=str, default=config.SAVE_DECODER_PATH, help='Filename to save the trained decoder model.')
       
    # Optional arguments for loading and loading model parameters
    parser.add_argument('-le', '--load_encoder', type=str, help='Filename to load encoder parameters from.')
    parser.add_argument('-ld', '--load_decoder', type=str, help='Filename to load decoder parameters from.')

    # Optional arguments for evaluating encoder and decoder
    parser.add_argument('-ee', '--evaluate_encoder', type=str, help='Filename for outputting encoder evaluation results.')
    parser.add_argument('-ed', '--evaluate_decoder', type=str, help='Filename for outputting decoder evaluation results.')

    # Optional arguments for selecting plugins
    parser.add_argument('-ep', '--encoder_plugin', type=str, default=config.DEFAULT_ENCODER_PLUGIN, help='Name of the encoder plugin to use.')
    parser.add_argument('-dp', '--decoder_plugin', type=str, default=config.DEFAULT_DECODER_PLUGIN, help='Name of the decoder plugin to use.')

    # Optional argument for specifying the sliding window size
    parser.add_argument('-ws', '--window_size', type=int, default=config.WINDOW_SIZE, help='Sliding window size to use for processing time series data.')
    
    # Optional arguments related to autoencoder configuration and training
    parser.add_argument('-me', '--max_error', type=float, default=config.MAXIMUM_MSE_THRESHOLD, help='Maximum MSE error to stop the training process.')
    parser.add_argument('-is', '--initial_size', type=int, default=config.INITIAL_ENCODING_DIM, help='Initial size of the encoder/decoder interface.')
    parser.add_argument('-ss', '--step_size', type=int, default=config.ENCODING_STEP_SIZE, help='Step size to reduce the size of the encoder/decoder interface on each iteration.')

    # Optional argument for remote logging, monitoring and storage of trained models
    parser.add_argument('-rl', '--remote_log', type=str, default=config.REMOTE_LOG_URL, help='URL of a remote data-logger API endpoint.')

    # Optional argument for downloading and executing a remote JSON configuration
    parser.add_argument('-rc', '--remote_config', type=str, default=config.REMOTE_CONFIG_URL, help='URL of a remote JSON configuration file to download and execute.')

    # Optional argument for loading local config file
    parser.add_argument('-lc', '--load_config', type=str, help='Path to a local JSON configuration file to load and execute.')

    # Optional argument for quiet mode
    parser.add_argument('-qm', '--quiet_mode', type=bool, default=config.DEFAULT_QUIET_MODE, help='Do not show results on console.')

    args, unknown = parser.parse_known_args()
    return args, unknown
