import argparse
from app.config import DEFAULT_VALUES

def parse_args():
    parser = argparse.ArgumentParser(description="Feature-extractor: A tool for encoding and decoding CSV data with support for dynamic plugins.")

    parser.add_argument('csv_file', type=str, help='Path to the CSV file to process.')
    parser.add_argument('-se', '--save_encoder', type=str, default=DEFAULT_VALUES['save_encoder_path'], help='Filename to save the trained encoder model.')
    parser.add_argument('-sd', '--save_decoder', type=str, default=DEFAULT_VALUES['save_decoder_path'], help='Filename to save the trained decoder model.')
    parser.add_argument('-le', '--load_encoder', type=str, help='Filename to load encoder parameters from.')
    parser.add_argument('-ld', '--load_decoder', type=str, help='Filename to load decoder parameters from.')
    parser.add_argument('-ee', '--evaluate_encoder', type=str, help='Filename for outputting encoder evaluation results.')
    parser.add_argument('-ed', '--evaluate_decoder', type=str, help='Filename for outputting decoder evaluation results.')
    parser.add_argument('-ep', '--encoder_plugin', type=str, default=DEFAULT_VALUES['encoder_plugin'], help='Name of the encoder plugin to use.')
    parser.add_argument('-dp', '--decoder_plugin', type=str, default=DEFAULT_VALUES['decoder_plugin'], help='Name of the decoder plugin to use.')
    parser.add_argument('-ws', '--window_size', type=int, default=DEFAULT_VALUES['window_size'], help='Sliding window size to use for processing time series data.')
    parser.add_argument('-me', '--max_error', type=float, default=DEFAULT_VALUES['maximum_mse_threshold'], help='Maximum MSE error to stop the training process.')
    parser.add_argument('-is', '--initial_size', type=int, default=DEFAULT_VALUES['initial_encoding_dim'], help='Initial size of the encoder/decoder interface.')
    parser.add_argument('-ss', '--step_size', type=int, default=DEFAULT_VALUES['encoding_step_size'], help='Step size to reduce the size of the encoder/decoder interface on each iteration.')
    parser.add_argument('-rl', '--remote_log', type=str, default=DEFAULT_VALUES['remote_log_url'], help='URL of a remote data-logger API endpoint.')
    parser.add_argument('-rc', '--remote_config', type=str, default=DEFAULT_VALUES['remote_config_url'], help='URL of a remote JSON configuration file to download and execute.')
    parser.add_argument('-qm', '--quiet_mode', type=bool, default=DEFAULT_VALUES['quiet_mode'], help='Do not show results on console.')
    parser.add_argument('-fd', '--force_date', type=bool, default=DEFAULT_VALUES['force_date'], help='Force date inclusion in output.')

    args, unknown = parser.parse_known_args()
    return args, unknown
