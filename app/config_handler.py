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
    print(f"Pre-Merge: file config: {config}")
    print(f"Pre-Merge: cli_args: {cli_args}")
    
    merged_config = DEFAULT_VALUES.copy()
    merged_config.update(config)
    merged_config.update({k: v for k, v in cli_args.items() if v is not None})
    
    # Handle unknown arguments and compare them with plugin-specific parameters
    encoder_plugin_params = encoder_plugin.plugin_params
    decoder_plugin_params = decoder_plugin.plugin_params

    print(f"Encoder plugin params before merging: {encoder_plugin_params}")
    print(f"Decoder plugin params before merging: {decoder_plugin_params}")

    for key, value in unknown_args.items():
        if key in encoder_plugin_params:
            encoder_plugin_params[key] = type(encoder_plugin_params[key])(value)
        if key in decoder_plugin_params:
            decoder_plugin_params[key] = type(decoder_plugin_params[key])(value)

    print(f"Encoder plugin params after merging: {encoder_plugin_params}")
    print(f"Decoder plugin params after merging: {decoder_plugin_params}")

    merged_config.update(encoder_plugin_params)
    merged_config.update(decoder_plugin_params)
    
    print(f"Post-Merge: {merged_config}")
    return merged_config

def configure_with_args(config, args):
    config.update(args)
    return config

def save_debug_info(debug_info, path='debug_out.json'):
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
