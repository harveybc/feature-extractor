# Configuration file for the feature-extractor application

# Default configuration values
DEFAULT_VALUES = {
    'csv_input_path': './csv_input.csv',
    'csv_output_path': './csv_output.csv',
    'config_save_path': './config_out.json',
    'config_load_path': './config_in.json',
    'encoder_plugin': 'default',
    'decoder_plugin': 'default',
    'training_batch_size': 64,  # Updated batch size
    'epochs': 20,  # Updated number of epochs
    'plugin_directory': 'app/plugins/',
    'remote_log_url': None,
    'remote_config_url': None,
    'window_size': 512,
    'initial_encoding_dim': 256,
    'encoding_step_size': 32,
    'mse_threshold': 0.3,  # Renamed parameter
    'quiet_mode': False,
    'remote_username': 'test',
    'remote_password': 'pass',
    'save_encoder_path': './encoder_ann.keras',
    'save_decoder_path': './decoder_ann.keras',
    'force_date': False,
    'headers': False,  # Ensure this default value is set
    'incremental_search': False  # New parameter for incremental search
}
