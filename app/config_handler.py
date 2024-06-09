import json
import requests
from app.config import DEFAULT_VALUES

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
