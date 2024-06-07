# Configuration file for the feature-extractor application

# Path configurations
CSV_INPUT_PATH = './csv_input.csv'  # Default path for CSV input if not specified
MODEL_SAVE_PATH = './model_out'          # Default directory to save trained models
MODEL_LOAD_PATH = './model_in'          # Default directory to load models from
CSV_OUTPUT_PATH = './csv_output.csv'  # Default path for CSV output if not specified
CONFIG_SAVE_PATH = './config_out.json'  # Default file to save preprocessing configurations
CONFIG_LOAD_PATH = './config_in.json'  # Default file to load configurations from

# Model and training configurations
DEFAULT_ENCODER_PLUGIN = 'basic_encoder'         # Default encoder plugin name
DEFAULT_DECODER_PLUGIN = 'basic_decoder'         # Default decoder plugin name
TRAINING_BATCH_SIZE = 32                         # Default batch size for model training
EPOCHS = 10                                      # Default number of epochs for training

# Plugin configurations
PLUGIN_DIRECTORY = 'app/plugins/'  # Directory containing all plugins

# Remote logging and configuration
REMOTE_LOG_URL = 'http://remote-log-server/api/logs'  # Default URL for remote logging
REMOTE_CONFIG_URL = 'http://remote-config-server/api/config'  # Default URL for remote configuration

# Sliding window size for time series processing
WINDOW_SIZE = 512                                # Default window size for sliding window technique

# Autoencoder configuration
INITIAL_ENCODING_DIM = 256                       # Initial encoding dimension for the encoder
ENCODING_STEP_SIZE = 32                          # Step size to reduce the encoding dimension
MAXIMUM_MSE_THRESHOLD = 0.01                     # Maximum MSE threshold for stopping training

# Quiet mode
DEFAULT_QUIET_MODE = False  # Default setting for quiet mode
