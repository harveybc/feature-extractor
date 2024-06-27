import json
import requests
from app.config import DEFAULT_VALUES

def load_config(file_path):
    print(f"Loading configuration from file: {file_path}")
    with open(file_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration: {config}")
    return config

def save_config(config, path='config_out.json'):
    config_to_save = {k: v for k, v in config.items() if k not in DEFAULT_VALUES or v != DEFAULT_VALUES[k]}
    print(f"Saving configuration to file: {path}")
    print(f"Configuration to save: {config_to_save}")
    with open(path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    return config, path

def merge_config(config, cli_args, unknown_args, encoder_plugin, decoder_plugin):
    print(f"Pre-Merge: default config: {DEFAULT_VALUES}")
    print(f"Pre-Merge: plugin default params: {encoder_plugin.plugin_params}, {decoder_plugin.plugin_params}")
    print(f"Pre-Merge: file config: {config}")
    print(f"Pre-Merge: cli_args: {cli_args}")

    # Start with default values from config.py
    merged_config = DEFAULT_VALUES.copy()
    print(f"Step 1 - Default config: intermediate_layers = {merged_config.get('intermediate_layers')}")

    # Merge with configuration from file
    merged_config.update(config)
    print(f"Step 2 - File config merged: intermediate_layers = {merged_config.get('intermediate_layers')}")

    # Merge with plugin default parameters
    for param, value in encoder_plugin.plugin_params.items():
        if param not in merged_config:
            merged_config[param] = value
    for param, value in decoder_plugin.plugin_params.items():
        if param not in merged_config:
            merged_config[param] = value
    print(f"Step 3 - Plugin defaults merged: intermediate_layers = {merged_config.get('intermediate_layers')}")

    # Filter out CLI arguments that were not explicitly set by the user
    cli_args_filtered = {k: v for k, v in cli_args.items() if v not in (None, False, '')}
    print(f"CLI arguments to merge: {cli_args_filtered}")

    # Update merged_config with filtered CLI arguments
    merged_config.update(cli_args_filtered)
    print(f"Step 4 - CLI arguments merged: intermediate_layers = {merged_config.get('intermediate_layers')}")

    # Set plugin parameters
    encoder_plugin.set_params(**{k: v for k, v in merged_config.items() if k in encoder_plugin.plugin_params})
    decoder_plugin.set_params(**{k: v for k, v in merged_config.items() if k in decoder_plugin.plugin_params})

    print(f"Final merged config: {merged_config}")
    print(f"Final merged config: intermediate_layers = {merged_config.get('intermediate_layers')}")

    return merged_config

def configure_with_args(config, args):
    config.update(args)
    return config

def save_debug_info(debug_info, encoder_plugin, decoder_plugin, path='debug_out.json'):
    encoder_debug_info = encoder_plugin.get_debug_info()
    decoder_debug_info = decoder_plugin.get_debug_info()
    
    debug_info = {
        'execution_time': debug_info.get('execution_time', 0),
        'encoder': encoder_debug_info,
        'decoder': decoder_debug_info
    }

    print(f"Saving debug information to file: {path}")
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)
    print(f"Debug information saved to {path}")

def load_remote_config(url, username, password):
    print(f"Loading remote configuration from URL: {url}")
    response = requests.get(url, auth=(username, password))
    response.raise_for_status()
    config = response.json()
    print(f"Loaded remote configuration: {config}")
    return config

def save_remote_config(config, url, username, password):
    print(f"Saving configuration to remote URL: {url}")
    response = requests.post(url, auth=(username, password), json=config)
    response.raise_for_status()
    success = response.status_code == 200
    print(f"Configuration saved to remote URL: {success}")
    return success

def log_remote_data(data, url, username, password):
    print(f"Logging data to remote URL: {url}")
    response = requests.post(url, auth=(username, password), json=data)
    response.raise_for_status()
    success = response.status_code == 200
    print(f"Data logged to remote URL: {success}")
    return success
