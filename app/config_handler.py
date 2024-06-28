import json
import sys
from app.config import DEFAULT_VALUES

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

def merge_config(config, cli_args, unknown_args, encoder_plugin, decoder_plugin):
    # Step 1: Start with default values from config.py
    merged_config = DEFAULT_VALUES.copy()

    # Step 2: Merge with plugin default parameters
    for k, v in encoder_plugin.plugin_params.items():
        merged_config[k] = v
    for k, v in decoder_plugin.plugin_params.items():
        merged_config[k] = v

    # Step 3: Merge with file configuration
    for k, v in config.items():
        print(f"Merging from file config: {k} = {v}")
        merged_config[k] = v
    print(f"Step 3 - File config merged: {merged_config}")
    desired_step3_output = {
        'csv_file': './csv_input.csv', 'save_encoder': './encoder_model.h5', 'save_decoder': './decoder_model.h5',
        'load_encoder': None, 'load_decoder': None, 'evaluate_encoder': './encoder_eval.csv',
        'evaluate_decoder': './decoder_eval.csv', 'encoder_plugin': 'default', 'decoder_plugin': 'default',
        'window_size': 128, 'threshold_error': 0.0003, 'initial_size': 8, 'step_size': 4,
        'remote_log': None, 'remote_config': None, 'load_config': './config_in.json', 'save_config': './config_out.json',
        'quiet_mode': False, 'force_date': False, 'incremental_search': True, 'headers': False, 'epochs': 5,
        'batch_size': 256, 'intermediate_layers': 2, 'layer_size_divisor': 2
    }
    print(f"Desired Step 3 Output: {desired_step3_output}")
    if merged_config != desired_step3_output:
        print(f"Actual Step 3 Output: {merged_config}")
        print("Error: Step 3 output does not match the desired output.")
        sys.exit(1)

    # Step 4: Identify and merge user-set CLI parameters
    user_set_cli_args = {}
    cli_arg_keys = [arg.lstrip('--') for arg in sys.argv[1:] if arg.startswith('--')]
    for key in cli_arg_keys:
        arg_key = key.split('=')[0]
        if arg_key in cli_args:
            user_set_cli_args[arg_key] = cli_args[arg_key]
            print(f"User set CLI argument: {arg_key} = {cli_args[arg_key]}")

    for key, value in user_set_cli_args.items():
        if value is not None:
            print(f"Merging from CLI args: {key} = {value}")
            merged_config[key] = value
    print(f"Step 4 - CLI arguments merged: {merged_config}")
    desired_step4_output = {
        'csv_file': 'tests\\data\\csv_sel_unb_norm_512.csv', 'save_encoder': './encoder_model.h5', 'save_decoder': './decoder_model.h5',
        'load_encoder': None, 'load_decoder': None, 'evaluate_encoder': './encoder_eval.csv', 'evaluate_decoder': './decoder_eval.csv',
        'encoder_plugin': 'lstm', 'decoder_plugin': 'lstm', 'window_size': 32, 'threshold_error': 0.0003, 'initial_size': 4,
        'step_size': 4, 'remote_log': None, 'remote_config': None, 'load_config': 'input_config.json', 'save_config': './config_out.json',
        'quiet_mode': False, 'force_date': False, 'incremental_search': False, 'headers': False, 'epochs': 5,
        'batch_size': 256, 'intermediate_layers': 4, 'layer_size_divisor': 2
    }
    print(f"Desired Step 4 Output: {desired_step4_output}")
    print(f"Actual Step 4 Output: {merged_config}")
    if merged_config != desired_step4_output:
        print("Error: Step 4 output does not match the desired output.")
        sys.exit(1)

    final_config = {}
    for k, v in merged_config.items():
        if k in config or k in cli_args or k in DEFAULT_VALUES:
            final_config[k] = v

    return final_config

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
