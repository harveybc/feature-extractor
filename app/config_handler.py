# config_handler.py

import json
import sys
import requests
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def get_plugin_default_params(plugin_name, plugin_type):
    plugin_class, _ = load_plugin(plugin_type, plugin_name)
    plugin_instance = plugin_class()
    return plugin_instance.plugin_params

def compose_config(config):
    encoder_plugin_name = config.get('encoder_plugin', DEFAULT_VALUES.get('encoder_plugin'))
    decoder_plugin_name = config.get('decoder_plugin', DEFAULT_VALUES.get('decoder_plugin'))
    
    encoder_default_params = get_plugin_default_params(encoder_plugin_name, 'feature_extractor.encoders')
    decoder_default_params = get_plugin_default_params(decoder_plugin_name, 'feature_extractor.decoders')

    config_to_save = {}
    for k, v in config.items():
        if k not in DEFAULT_VALUES or v != DEFAULT_VALUES[k]:
            if k not in encoder_default_params or v != encoder_default_params[k]:
                if k not in decoder_default_params or v != decoder_default_params[k]:
                    config_to_save[k] = v
    
    # prints config_to_save
    print(f"Actual config_to_save: {config_to_save}")
    return config_to_save

def save_config(config, path='config_out.json'):
    config_to_save = compose_config(config)
    
    with open(path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    return config, path

def save_debug_info(debug_info, path='debug_out.json'):
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)

def remote_save_config(config, url, username, password):
    config_to_save = compose_config(config)
    try:
        response = requests.post(
            url,
            auth=(username, password),
            data={'json_config': json.dumps(config_to_save)}
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Failed to save remote configuration: {e}", file=sys.stderr)
        return False
    
def remote_load_config(url, username=None, password=None):
    try:
        if username and password:
            response = requests.get(url, auth=(username, password))
        else:
            response = requests.get(url)
        response.raise_for_status()
        config = response.json()
        return config
    except requests.RequestException as e:
        print(f"Failed to load remote configuration: {e}", file=sys.stderr)
        return None

def remote_log(config, debug_info, url, username, password):
    config_to_save = compose_config(config)
    try:
        data = {
            'json_config': json.dumps(config_to_save),
            'json_result': json.dumps(debug_info)
        }
        response = requests.post(
            url,
            auth=(username, password),
            data=data
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Failed to log remote information: {e}", file=sys.stderr)
        return False
