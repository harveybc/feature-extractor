import json
import requests
from app.config import DEFAULT_VALUES
from config_merger import merge_config

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, path='config_out.json'):
    config_to_save = {}
    for k, v in config.items():
        if k not in DEFAULT_VALUES or v != DEFAULT_VALUES[k]:
            config_to_save[k] = v
    with open(path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    return config, path

def configure_with_args(config, args):
    for k, v in args.items():
        config[k] = v
    return config

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

# Usage example
if __name__ == "__main__":
    # Load default values
    config = DEFAULT_VALUES.copy()

    # Load configuration from file
    file_config = load_config(config['load_config'])

    # Example plugin defaults (encoder and decoder)
    encoder_plugin_defaults = {'epochs': 10, 'batch_size': 256, 'intermediate_layers': 1, 'layer_size_divisor': 2}
    decoder_plugin_defaults = {'epochs': 10, 'batch_size': 256, 'intermediate_layers': 1, 'layer_size_divisor': 2}

    # Parse CLI arguments
    cli_args = {
        'csv_file': 'tests\\data\\csv_sel_unb_norm_512.csv',
        'encoder_plugin': 'cnn',
        'decoder_plugin': 'cnn',
        'window_size': 32,
        'initial_size': 4,
        'load_config': 'input_config.json',
        'intermediate_layers': 4
    }

    # Merge configurations
    final_config = merge_config(DEFAULT_VALUES, {**encoder_plugin_defaults, **decoder_plugin_defaults}, file_config, cli_args)
    print(f"Final merged config: {final_config}")
