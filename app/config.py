# Configuration file for the feature-extractor application

DEFAULT_VALUES = {
    # Path configurations
    'csv_input_path': './csv_input.csv',  # Default path for CSV input if not specified
    'csv_output_path': './csv_output.csv',  # Default path for CSV output if not specified
    'config_save_path': './config_out.json',  # Default file to save preprocessing configurations
    'config_load_path': './config_in.json',  # Default file to load configurations from

    # Model and training configurations
    'encoder_plugin': 'default',  # Default encoder plugin name
    'decoder_plugin': 'default',  # Default decoder plugin name
    'training_batch_size': 32,  # Default batch size for model training
    'epochs': 10,  # Default number of epochs for training

    # Plugin configurations
    'plugin_directory': 'app/plugins/',  # Directory containing all plugins

    # Remote logging and configuration
    'remote_log_url': None,  # Default URL for remote logging
    'remote_config_url': None,  # Default URL for remote configuration

    # Sliding window size for time series processing
    'window_size': 512,  # Default window size for sliding window technique

    # Autoencoder configuration
    'initial_encoding_dim': 256,  # Initial encoding dimension for the encoder
    'encoding_step_size': 32,  # Step size to reduce the encoding dimension
    'maximum_mse_threshold': 0.3,  # Maximum MSE threshold for stopping training

    # Quiet mode
    'quiet_mode': False,  # Default setting for quiet mode

    # Default remote credentials
    'remote_username': 'test',
    'remote_password': 'pass',

    # Save paths for encoder and decoder
    'save_encoder_path': './encoder_ann.keras',  # Default path to save the encoder model
    'save_decoder_path': './decoder_ann.keras',  # Default path to save the decoder model

    # Force date inclusion
    'force_date': False  # Default setting for force date inclusion in output
}
