# Configuration file for the feature-extractor application

# Default configuration values
DEFAULT_VALUES = {
    'csv_input_path': './csv_input.csv',
    'csv_output_path': './csv_output.csv',
    'config_save_path': './config_out.json',
    'config_load_path': './config_in.json',
    'encoder_plugin': 'default',
    'decoder_plugin': 'default',
    'training_batch_size': 32,
    'epochs': 10,
    'plugin_directory': 'app/plugins/',
    'remote_log_url': None,
    'remote_config_url': None,
    'window_size': 512,
    'initial_encoding_dim': 256,
    'encoding_step_size': 32,
    'maximum_mse_threshold': 0.3,
    'quiet_mode': False,
    'remote_username': 'test',
    'remote_password': 'pass',
    'save_encoder_path': './encoder_ann.keras',
    'save_decoder_path': './decoder_ann.keras',
    'force_date': False,
    'headers': False  # Ensure this default value is set
}
