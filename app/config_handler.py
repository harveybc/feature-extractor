# config_handler.py

import json
import os
import sys
import requests
import numpy as np # Import numpy
import tensorflow as tf # For tf.print
import datetime # For NpEncoder
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

def _sanitize_value(value):
    """
    Recursively sanitizes a value to make it JSON-serializable,
    replacing large objects with placeholders.
    """
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    elif isinstance(value, np.ndarray):
        return f"<ndarray shape:{value.shape} dtype:{value.dtype} removed>"
    elif isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, tf.Tensor):
        return f"<Tensor shape:{value.shape} dtype:{value.dtype.name} removed>"
    # Add more specific types if needed, e.g., pandas DataFrame/Series
    elif hasattr(value, 'to_dict'): # Basic check for pandas-like objects
        try:
            # If it's small, maybe allow, but generally safer to summarize
            return f"<pandas-like object type:{type(value).__name__} removed>"
        except Exception:
            return f"<pandas-like object type:{type(value).__name__} (error converting) removed>"
    else:
        # For other complex objects, return a type placeholder
        return f"<complex_object type:{type(value).__name__} removed>"

def sanitize_dict_for_json(input_dict):
    """
    Creates a sanitized deep copy of a dictionary suitable for JSON serialization.
    """
    if not isinstance(input_dict, dict):
        tf.print(f"Warning: sanitize_dict_for_json expects a dict, got {type(input_dict)}. Returning placeholder.")
        return {"error": f"Expected dict, got {type(input_dict)}"}
    
    # Perform a deep-ish copy for common structures, then sanitize
    # For true deep copy of arbitrary objects, copy.deepcopy might be needed,
    # but we are aiming to simplify, not perfectly replicate.
    copied_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, (dict, list, tuple)):
            # This is a shallow copy for the first level of dict/list/tuple,
            # _sanitize_value will handle recursion.
            copied_dict[k] = v 
        else:
            copied_dict[k] = v
            
    return _sanitize_value(copied_dict)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray): # Should be caught by sanitize_dict_for_json
            return f"<ndarray shape:{obj.shape} dtype:{obj.dtype} (NpEncoder fallback)>"
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, tf.Tensor): # Should be caught by sanitize_dict_for_json
            return f"<Tensor shape:{obj.shape} dtype:{obj.dtype.name} (NpEncoder fallback)>"
        return super(NpEncoder, self).default(obj)

def save_config_to_file(config, file_path):
    """Saves the configuration dictionary to a JSON file after sanitizing."""
    try:
        # Sanitize a DEEP COPY of the config to avoid modifying the original
        config_to_save = sanitize_dict_for_json(config) # Apply sanitization
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config_to_save, f, indent=4, cls=NpEncoder)
        tf.print(f"Sanitized configuration saved to {file_path}")
    except Exception as e:
        tf.print(f"Error saving sanitized configuration to {file_path}: {e}")
        import traceback
        tf.print(traceback.format_exc())

def save_debug_info(debug_info, file_path):
    """Saves the debug information dictionary to a JSON file."""
    # Assuming debug_info is already constructed with sanitized/filtered data by data_processor.py
    try:
        # Even so, apply sanitization as a final safeguard
        debug_info_to_save = sanitize_dict_for_json(debug_info)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(debug_info_to_save, f, indent=4, cls=NpEncoder)
        tf.print(f"Sanitized debug info saved to {file_path}")
    except Exception as e:
        tf.print(f"Error saving sanitized debug info to {file_path}: {e}")
        import traceback
        tf.print(traceback.format_exc())

def remote_log(config_arg, data, url, username, password): # Renamed first arg to avoid confusion
    """Placeholder for remote logging. Sanitizes config before 'sending'. """
    tf.print(f"Attempting to remote log to {url} for user {username}.")
    try:
        sanitized_config_for_remote = sanitize_dict_for_json(config_arg)
        # In a real scenario, you would serialize `data` (which is debug_info)
        # and `sanitized_config_for_remote` and send them.
        # For now, just print that it would happen.
        # remote_log_actual(sanitized_config_for_remote, data, url, username, password)
        tf.print(f"Remote logging: Config (sanitized) and data (debug_info) would be sent.")
        tf.print(f"Sanitized config for remote log includes keys: {list(sanitized_config_for_remote.keys())}")
        tf.print(f"Debug info for remote log includes keys: {list(data.keys())}")
    except Exception as e:
        tf.print(f"Error during remote_log preparation: {e}")

# ... (rest of your config_handler.py, e.g., load_config_from_file, get_config)
# Ensure get_config returns a fresh copy or is not modified with large data.
# DEFAULT_VALUES should only contain simple types.

DEFAULT_VALUES = {
    # ... your existing default values, ensure NO numpy arrays here ...
    # Example:
    "learning_rate": 1e-4, # This is fine
    # "some_array_default": np.array([1,2,3]) # THIS IS BAD for DEFAULT_VALUES
}

# It's crucial that the 'config' object loaded and used throughout the application
# does not get large arrays assigned to its top-level keys if that same 'config' object
# is later passed to save_config_to_file.
# If data_processor adds things like 'cvae_val_inputs' directly to the main 'config' dict,
# then save_config_to_file(config, ...) will try to save them.
# The sanitize_dict_for_json is a strong defense, but good practice is to avoid
# polluting the main config dict with transient large data.
