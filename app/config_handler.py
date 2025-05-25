# config_handler.py

import json
import sys
import requests
import numpy as np # Import numpy
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin

def convert_numpy_types(obj):
    """
    Recursively converts NumPy types in a dictionary or list to Python native types
    for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    elif isinstance(obj, np.generic): # Catches np.float32, np.int64, np.bool_, etc.
        return obj.item()  # Convert NumPy scalar to Python native type
    return obj

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
        # Check if the key is in DEFAULT_VALUES and if the value is different
        is_default_value_changed = k in DEFAULT_VALUES and v != DEFAULT_VALUES[k]
        is_not_in_default_values = k not in DEFAULT_VALUES

        # Check if the key is in encoder_default_params and if the value is different
        is_encoder_default_param_changed = k in encoder_default_params and v != encoder_default_params[k]
        is_not_in_encoder_default_params = k not in encoder_default_params
        
        # Check if the key is in decoder_default_params and if the value is different
        is_decoder_default_param_changed = k in decoder_default_params and v != decoder_default_params[k]
        is_not_in_decoder_default_params = k not in decoder_default_params

        # Condition to save:
        # 1. Key is not in DEFAULT_VALUES OR its value is different from the default
        # AND
        # 2. Key is not in encoder_default_params OR its value is different from the encoder default
        # AND
        # 3. Key is not in decoder_default_params OR its value is different from the decoder default
        if (is_not_in_default_values or is_default_value_changed) and \
           (is_not_in_encoder_default_params or is_encoder_default_param_changed) and \
           (is_not_in_decoder_default_params or is_decoder_default_param_changed):
            config_to_save[k] = v
            
    # prints config_to_save
    print(f"Actual config_to_save (before numpy conversion): {config_to_save}")
    return config_to_save # Return potentially unconverted, conversion will happen before dump

def save_config(config, path='config_out.json'):
    config_composed = compose_config(config)
    config_to_save_serializable = convert_numpy_types(config_composed) # Convert before saving
    
    with open(path, 'w') as f:
        json.dump(config_to_save_serializable, f, indent=4)
    return config, path # Original config is returned, path to saved file

def save_debug_info(debug_info, path='debug_out.json'):
    debug_info_serializable = convert_numpy_types(debug_info) # Convert before saving
    with open(path, 'w') as f:
        json.dump(debug_info_serializable, f, indent=4)

def remote_save_config(config, url, username, password):
    config_composed = compose_config(config)
    config_to_save_serializable = convert_numpy_types(config_composed) # Convert before sending
    try:
        response = requests.post(
            url,
            auth=(username, password),
            data={'json_config': json.dumps(config_to_save_serializable)} # Use dumps for string
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
    config_composed = compose_config(config)
    config_to_save_serializable = convert_numpy_types(config_composed) # Convert config
    debug_info_serializable = convert_numpy_types(debug_info) # Convert debug_info
    try:
        data = {
            'json_config': json.dumps(config_to_save_serializable), # Use dumps for string
            'json_result': json.dumps(debug_info_serializable)      # Use dumps for string
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
