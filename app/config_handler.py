import json

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
    'window_size': 10,
    'max_error': 0.01,
    'initial_size': 256,
    'step_size': 32
}

def load_config(file_path):
    """
    Load configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {file_path} not found.")
        config = {}
    return config

def save_config(config, file_path):
    """
    Save configuration to a JSON file.

    Args:
        config (dict): Configuration dictionary to save.
        file_path (str): Path to the JSON configuration file.

    Returns:
        tuple: Configuration dictionary and file path.
    """
    config_to_save = {k: v for k, v in config.items() if k not in DEFAULT_VALUES or config[k] != DEFAULT_VALUES[k]}
    with open(file_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    return config, file_path

def merge_config(config, cli_args):
    """
    Merge CLI arguments into the configuration dictionary.

    Args:
        config (dict): Base configuration dictionary.
        cli_args (dict): CLI arguments to merge.

    Returns:
        dict: Merged configuration dictionary.
    """
    for key, value in DEFAULT_VALUES.items():
        config.setdefault(key, value)
    
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value

    return config


def save_debug_info(debug_info, path='debug_out.json'):
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)

def save_debug_info(debug_info, file_path):
    """
    Save debug information to a JSON file.

    Args:
        debug_info (dict): Debug information to save.
        file_path (str): Path to the JSON debug file.

    Returns:
        None
    """
    with open(file_path, 'w') as f:
        json.dump(debug_info, f, indent=4)

def load_remote_config(url=DEFAULT_VALUES['remote_load_config'], username=DEFAULT_VALUES['remote_username'], password=DEFAULT_VALUES['remote_password']):
    """
    Load configuration from a remote JSON file via HTTP GET request.

    Args:
        url (str): URL of the remote JSON configuration file.
        username (str): Username for HTTP authentication.
        password (str): Password for HTTP authentication.

    Returns:
        dict: Configuration dictionary.
    """
    response = requests.get(url, auth=(username, password))
    response.raise_for_status()
    return response.json()

def save_remote_config(config, url=DEFAULT_VALUES['remote_save_config'], username=DEFAULT_VALUES['remote_username'], password=DEFAULT_VALUES['remote_password']):
    """
    Save configuration to a remote JSON file via HTTP POST request.

    Args:
        config (dict): Configuration dictionary to save.
        url (str): URL of the remote API endpoint.
        username (str): Username for HTTP authentication.
        password (str): Password for HTTP authentication.

    Returns:
        dict: Response from the server.
    """
    response = requests.post(url, auth=(username, password), json=config)
    response.raise_for_status()
    return response.json()

def log_remote_data(data, url=DEFAULT_VALUES['remote_log'], username=DEFAULT_VALUES['remote_username'], password=DEFAULT_VALUES['remote_password']):
    """
    Log data to a remote server via HTTP POST request.

    Args:
        data (dict): Data to log.
        url (str): URL of the remote logging API endpoint.
        username (str): Username for HTTP authentication.
        password (str): Password for HTTP authentication.

    Returns:
        dict: Response from the server.
    """
    response = requests.post(url, auth=(username, password), json=data)
    response.raise_for_status()
    return response.json()