import json
import requests

# Define default values for the configuration
DEFAULT_VALUES = {
    'encoder_plugin': 'default_encoder',
    'decoder_plugin': 'default_decoder',
    'output_file': 'output.csv',
    'remote_log': 'http://localhost:60500/preprocessor/feature_extractor/create',
    'remote_save_config': 'http://localhost:60500/preprocessor/feature_extractor/create',
    'remote_load_config': 'http://localhost:60500/preprocessor/feature_extractor/detail/1',
    'remote_username': 'test',
    'remote_password': 'pass',
    'quiet_mode': False,
    'max_error': 0.01,
    'initial_size': 256,
    'step_size': 32,
    'csv_file': '',
    'headers': True,
    'window_size': 10
}

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, path='config_out.json'):
    config_to_save = {k: v for k, v in config.items() if k not in DEFAULT_VALUES or config[k] != DEFAULT_VALUES[k]}
    with open(path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    return config, path

def merge_config(config, cli_args, plugin_params):
    # Set default values
    for key, value in DEFAULT_VALUES.items():
        config.setdefault(key, value)

    # Merge CLI arguments, overriding config file values
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value

    # Merge plugin-specific arguments
    for key, value in plugin_params.items():
        config[key] = value

    return config

def save_debug_info(debug_info, path='debug_out.json'):
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)

def load_remote_config(url=DEFAULT_VALUES['remote_load_config'], username=DEFAULT_VALUES['remote_username'], password=DEFAULT_VALUES['remote_password']):
    response = requests.get(url, auth=(username, password))
    response.raise_for_status()
    return response.json()

def save_remote_config(config, url=DEFAULT_VALUES['remote_save_config'], username=DEFAULT_VALUES['remote_username'], password=DEFAULT_VALUES['remote_password']):
    response = requests.post(url, auth=(username, password), json=config)
    response.raise_for_status()
    return response.status_code == 200

def log_remote_data(data, url=DEFAULT_VALUES['remote_log'], username=DEFAULT_VALUES['remote_username'], password=DEFAULT_VALUES['remote_password']):
    response = requests.post(url, auth=(username, password), json=data)
    response.raise_for_status()
    return response.status_code == 200
