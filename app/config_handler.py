# config_handler.py

import json
import requests
from app.config import DEFAULT_VALUES
from app.plugins.encoder_plugin_cnn import Plugin as EncoderPluginCNN
from app.plugins.decoder_plugin_transformer import Plugin as DecoderPluginTransformer

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, path='config_out.json'):
    # Retrieve plugin-specific default values
    encoder_plugin_default_values = EncoderPluginCNN().plugin_params
    decoder_plugin_default_values = DecoderPluginTransformer().plugin_params
    
    config_to_save = {}
    for k, v in config.items():
        if k not in DEFAULT_VALUES or v != DEFAULT_VALUES[k]:
            # Check for plugin-specific default values
            if k in encoder_plugin_default_values and v == encoder_plugin_default_values[k]:
                continue
            if k in decoder_plugin_default_values and v == decoder_plugin_default_values[k]:
                continue
            config_to_save[k] = v
    with open(path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    return config, path

def save_debug_info(debug_info, encoder_plugin, decoder_plugin, path='debug_out.json'):
    encoder_debug_info = encoder_plugin.get_debug_info()
    decoder_debug_info = decoder_plugin.get_debug_info()
    
    debug_info = {
        'execution_time': debug_info.get('execution_time', 0),
        'encoder': encoder_debug_info,
        'decoder': decoder_debug_info
    }

    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)

def load_remote_config(url, username, password):
    response = requests.get(url, auth=(username, password))
    response.raise_for_status()
    config = response.json()
    return config

def save_remote_config(config, url, username, password):
    response = requests.post(url, auth=(username, password), json=config)
    response.raise_for_status()
    success = response.status_code == 200
    return success

def log_remote_data(data, url, username, password):
    response = requests.post(url, auth=(username, password), json=data)
    response.raise_for_status()
    success = response.status_code == 200
    return success
