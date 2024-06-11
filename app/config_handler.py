import json
import requests
from app.config import DEFAULT_VALUES

def load_config(file_path):
    """
    Load the configuration from a JSON file.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    print(f"Loading configuration from file: {file_path}")
    with open(file_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration: {config}")
    return config

def save_config(config, path='config_out.json'):
    """
    Save the configuration to a JSON file, excluding default values.

    Args:
        config (dict): The configuration to save.
        path (str): The path to the output configuration file.

    Returns:
        tuple: The saved configuration and the path to the file.
    """
    config_to_save = {k: v for k, v in config.items() if k not in DEFAULT_VALUES or config[k] != DEFAULT_VALUES[k]}
    print(f"Saving configuration to file: {path}")
    print(f"Configuration to save: {config_to_save}")
    with open(path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    return config, path

def merge_config(config, cli_args, plugin_params):
    """
    Merge the default configuration, file configuration, CLI arguments, and plugin parameters.

    Args:
        config (dict): The base configuration.
        cli_args (dict): The command-line arguments.
        plugin_params (dict): The plugin-specific parameters.

    Returns:
        dict: The merged configuration.
    """
    print(f"Pre-Merge: default config: {DEFAULT_VALUES}")
    print(f"Pre-Merge: file config: {config}")
    print(f"Pre-Merge: cli_args: {cli_args}")
    print(f"Pre-Merge: plugin_params: {plugin_params}")

