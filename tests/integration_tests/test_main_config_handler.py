import json
import requests
from app import config

# Define default values for the configuration
DEFAULT_VALUES = {
    'encoder_plugin': config.DEFAULT_ENCODER_PLUGIN,
    'decoder_plugin': config.DEFAULT_DECODER_PLUGIN,
    'output_file': config.CSV_OUTPUT_PATH,
    'remote_log': config.REMOTE_LOG_URL,
    'remote_save_config': config.REMOTE_CONFIG_URL,
    'remote_load_config': config.REMOTE_CONFIG_URL,
    'remote_username': config.REMOTE_USERNAME,
    'remote_password': config.REMOTE_PASSWORD,
    'quiet_mode': config.DEFAULT_QUIET_MODE,
    'max_error': config.MAXIMUM_MSE_THRESHOLD,
    'initial_size': config.INITIAL_ENCODING_DIM,
    'step_size': config.ENCODING_STEP_SIZE,
    'csv_file': config.CSV_INPUT_PATH,
    'headers': False,
    'window_size': config.WINDOW_SIZE,
    'save_encoder': config.SAVE_ENCODER_PATH,
    'save_decoder': config.SAVE_DECODER_PATH
}

def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_config(config, path='config_out.json'):
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
