# Configuration file for the feature-extractor application

# Path configurations
CSV_INPUT_PATH = 'path/to/default/csv_input.csv'  # Default path for CSV input if not specified
MODEL_SAVE_PATH = 'path/to/save/models/'          # Default directory to save trained models
MODEL_LOAD_PATH = 'path/to/load/models/'          # Default directory to load models from
OUTPUT_PATH = 'path/to/output/'                   # Default directory for saving outputs

# Model and training configurations
DEFAULT_ENCODER_PLUGIN = 'basic_encoder'         # Default encoder plugin name
DEFAULT_DECODER_PLUGIN = 'basic_decoder'         # Default decoder plugin name
TRAINING_BATCH_SIZE = 32                         # Default batch size for model training
EPOCHS = 10                                      # Default number of epochs for training

# Performance thresholds
MINIMUM_MSE_THRESHOLD = 0.01                     # Default minimum MSE for stopping training

# Plugin configurations
PLUGIN_DIRECTORY = 'app/plugins/'                # Directory containing all plugins

