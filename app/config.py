# Configuration file for the feature-extractor application

# Path configurations
CSV_INPUT_PATH = './csv_input.csv'  # Default path for CSV input if not specified
CSV_OUTPUT_PATH = './csv_output.csv'  # Default path for CSV output if not specified
CONFIG_SAVE_PATH = './config_out.json'  # Default file to save preprocessing configurations
CONFIG_LOAD_PATH = './config_in.json'  # Default file to load configurations from

# Model and training configurations
DEFAULT_ENCODER_PLUGIN = 'default'  # Default encoder plugin name
DEFAULT_DECODER_PLUGIN = 'default'  # Default decoder plugin name
TRAINING_BATCH_SIZE = 64  # Default batch size for model training
EPOCHS = 50  # Default number of epochs for training

# Plugin configurations
PLUGIN_DIRECTORY = 'app/plugins/'  # Directory containing all plugins

# Remote logging and configuration
REMOTE_LOG_URL = None  # Default URL for remote logging
REMOTE_CONFIG_URL = None  # Default URL for remote configuration

# Sliding window size for time series processing
WINDOW_SIZE = 512  # Default window size for sliding window technique

# Autoencoder configuration
INITIAL_ENCODING_DIM = 256  # Initial encoding dimension for the encoder
ENCODING_STEP_SIZE = 32  # Step size to adjust the encoding dimension
MSE_THRESHOLD = 0.3  # MSE threshold for stopping training
INCREMENTAL_SEARCH = False  # Default setting for incremental search

# Quiet mode
DEFAULT_QUIET_MODE = False  # Default setting for quiet mode

# Default remote credentials
REMOTE_USERNAME = 'test'
REMOTE_PASSWORD = 'pass'

# Save paths for encoder and decoder
SAVE_ENCODER_PATH = './encoder_ann.keras'  # Default path to save the encoder model
SAVE_DECODER_PATH = './decoder_ann.keras'  # Default path to save the decoder model
