# Configuration file for the feature-extractor application

# Default configuration values
DEFAULT_VALUES = {
    'csv_file': './csv_input.csv',                   # Default path for input CSV
    'save_encoder': './encoder_model.h5',            # Default filename to save the trained encoder model
    'save_decoder': './decoder_model.h5',            # Default filename to save the trained decoder model
    'load_encoder': None,                            # Default filename to load encoder parameters from
    'load_decoder': None,                            # Default filename to load decoder parameters from
    'evaluate_encoder': './encoder_eval.csv',        # Default filename for outputting encoder evaluation results
    'evaluate_decoder': './decoder_eval.csv',        # Default filename for outputting decoder evaluation results
    'encoder_plugin': 'default',                     # Default encoder plugin to use
    'decoder_plugin': 'default',                     # Default decoder plugin to use
    'window_size': 512,                              # Default sliding window size for processing time series data
    'threshold_error': 0.005,                        # Default MSE error threshold to stop the training processes
    'initial_size': 8,                               # Default initial size of the encoder/decoder interface
    'step_size': 4,                                  # Default step size to adjust the size of the encoder/decoder interface
    'remote_log': None,                              # Default URL of a remote data-logger API endpoint
    'remote_config': None,                           # Default URL of a remote JSON configuration file to download and execute
    'load_config': './config_in.json',               # Default path to load a configuration file
    'save_config': './config_out.json',              # Default path to save the current configuration
    'quiet_mode': False,                             # Default setting to suppress output messages
    'force_date': False,                             # Default setting to include date in the output CSV files
    'incremental_search': True                       # Default setting to enable incremental search for interface size
}
