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
    with open(file_path, 'r') as f:
        return json.load(f)

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
    print(f"Pre-Merge: config: {config}")
    print(f"Pre-Merge: cli_args: {cli_args}")
    print(f"Pre-Merge: plugin_params: {plugin_params}")
    merged_config = {**DEFAULT_VALUES, **config, **plugin_params, **cli_args}
    print(f"Post-Merge: {merged_config}")
    return merged_config

def save_debug_info(debug_info, path='debug_out.json'):
    """
    Save debugging information to a JSON file.

    Args:
        debug_info (dict): The debugging information to save.
        path (str): The path to the output debug file.
    """
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)

def load_remote_config(url, username, password):
    """
    Load the configuration from a remote URL.

    Args:
        url (str): The URL to load the configuration from.
        username (str): The username for authentication.
        password (str): The password for authentication.

    Returns:
        dict: The loaded remote configuration as a dictionary.
    """
    response = requests.get(url, auth=(username, password))
    response.raise_for_status()
    return response.json()

def save_remote_config(config, url, username, password):
    """
    Save the configuration to a remote URL.

    Args:
        config (dict): The configuration to save.
        url (str): The URL to save the configuration to.
        username (str): The username for authentication.
        password (str): The password for authentication.

    Returns:
        bool: True if the configuration was successfully saved.
    """
    response = requests.post(url, auth=(username, password), json=config)
    response.raise_for_status()
    return response.status_code == 200

def log_remote_data(data, url, username, password):
    """
    Log data to a remote URL.

    Args:
        data (dict): The data to log.
        url (str): The URL to log the data to.
        username (str): The username for authentication.
        password (str): The password for authentication.

    Returns:
        bool: True if the data was successfully logged.
    """
    response = requests.post(url, auth=(username, password), json=data)
    response.raise_for_status()
    return response.status_code == 200
