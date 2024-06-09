import json
import requests
from config import (DEFAULT_ENCODER_PLUGIN, DEFAULT_DECODER_PLUGIN, 
                    CSV_OUTPUT_PATH, REMOTE_LOG_URL, REMOTE_CONFIG_URL, 
                    REMOTE_USERNAME, REMOTE_PASSWORD, DEFAULT_QUIET_MODE,
                    MAXIMUM_MSE_THRESHOLD, INITIAL_ENCODING_DIM, 
                    ENCODING_STEP_SIZE, SAVE_ENCODER_PATH, SAVE_DECODER_PATH, 
                    WINDOW_SIZE, CONFIG_SAVE_PATH, CONFIG_LOAD_PATH)

# Define default values for the configuration
DEFAULT_VALUES = {
    'encoder_plugin': DEFAULT_ENCODER_PLUGIN,
    'decoder_plugin': DEFAULT_DECODER_PLUGIN,
    'output_file': CSV_OUTPUT_PATH,
    'remote_log': REMOTE_LOG_URL,
    'remote_save_config': REMOTE_CONFIG_URL,
    'remote_load_config': REMOTE_CONFIG_URL,
    'remote_username': REMOTE_USERNAME,
    'remote_password': REMOTE_PASSWORD,
    'quiet_mode': DEFAULT_QUIET_MODE,
    'max_error': MAXIMUM_MSE_THRESHOLD,
    'initial_size': INITIAL_ENCODING_DIM,
    'step_size': ENCODING_STEP_SIZE,
    'csv_file': '',
    'headers': False,
    'window_size': WINDOW_SIZE,
    'save_encoder': SAVE_ENCODER_PATH,
    'save_decoder': SAVE_DECODER_PATH,
    'load_encoder': None,
    'load_decoder': None,
    'evaluate_encoder': None,
    'evaluate_decoder': None,
    'remote_config': None,
    'load_config': CONFIG_LOAD_PATH
}

def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_config(config, path=CONFIG_SAVE_PATH):
    config_to_save = {k: v for k, v in config.items() if k not in DEFAULT_VALUES or config[k] != DEFAULT_VALUES[k]}
    with open(path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    return config, path

def merge_config(config, cli_args, plugin_params):
    merged_config = {**DEFAULT_VALUES, **config, **cli_args, **plugin_params}
    return merged_config

def save_debug_info(debug_info, path='debug_out.json'):
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)

def load_remote_config(url, username, password):
    response = requests.get(url, auth=(username, password))
    response.raise_for_status()
    return response.json()

def save_remote_config(config, url, username, password):
    response = requests.post(url, auth=(username, password), json=config)
    response.raise_for_status()
    return response.status_code == 200

def log_remote_data(data, url, username, password):
    response = requests.post(url, auth=(username, password), json=data)
    response.raise_for_status()
    return response.status_code == 200
