import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Feature-extractor: A tool for encoding and decoding CSV data with support for dynamic plugins.")
    parser.add_argument('--input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('--validation_file', type=str, help='Path to the input CSV file uset do test the trained autoencoder.')
    
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file.')
    parser.add_argument('--save_encoder', type=str, help='Filename to save the trained encoder model.')
    parser.add_argument('--save_decoder', type=str, help='Filename to save the trained decoder model.')
    parser.add_argument('--load_encoder', type=str, help='Filename to load encoder parameters from.')
    parser.add_argument('--load_decoder', type=str, help='Filename to load decoder parameters from.')
    parser.add_argument('--evaluate_encoder', type=str, help='Filename for outputting encoder evaluation results.')
    parser.add_argument('--evaluate_decoder', type=str, help='Filename for outputting decoder evaluation results.')
    parser.add_argument('--encoder_plugin', type=str, help='Name of the encoder plugin to use.')
    parser.add_argument('--decoder_plugin', type=str, help='Name of the decoder plugin to use.')
    parser.add_argument('--window_size', type=int, help='Sliding window size to use for processing time series data.')
    parser.add_argument('--threshold_error', type=float, help='MSE error threshold to stop the training processes.')
    parser.add_argument('--initial_size', type=int, help='Initial size of the encoder/decoder interface.')
    parser.add_argument('--step_size', type=int, help='Step size to adjust the size of the encoder/decoder interface.')
    parser.add_argument('--remote_log', type=str, help='URL of a remote API endpoint for saving debug variables in JSON format.')
    parser.add_argument('--remote_load_config', type=str, help='URL of a remote JSON configuration file to download and execute.')
    parser.add_argument('--remote_save_config', type=str, help='URL of a remote API endpoint for saving configuration in JSON format.')
    parser.add_argument('--username', type=str, help='Username for the API endpoint.')
    parser.add_argument('--password', type=str, help='Password for Username for the API endpoint.')
    parser.add_argument('--load_config', type=str, help='Path to load a configuration file.')
    parser.add_argument('--save_config', type=str, help='Path to save the current configuration.')
    parser.add_argument('--save_log', type=str, help='Path to save the current debug info.')
    parser.add_argument('--quiet_mode', action='store_true', help='Suppress output messages.')
    parser.add_argument('--force_date', action='store_true', help='Include date in the output CSV files.')
    parser.add_argument('--incremental_search', action='store_true', help='Enable incremental search for interface size.')
    parser.add_argument('--headers', action='store_true', help='Indicate if the CSV file has headers.')
    parser.add_argument('--use_sliding_windows', action='store_true', help='Indicate if the CSV file has headers.')
    return parser.parse_known_args()
