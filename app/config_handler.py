import json
import requests
import sys
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

    # Step 1: Start with default values from config.py
    merged_config = DEFAULT_VALUES.copy()

    # Desired output for Step 1
    desired_step1_output = {
        'csv_file': './csv_input.csv',
        'save_encoder': './encoder_model.h5',
        'save_decoder': './decoder_model.h5',
        'load_encoder': None,
        'load_decoder': None,
        'evaluate_encoder': './encoder_eval.csv',
        'evaluate_decoder': './decoder_eval.csv',
        'encoder_plugin': 'default',
        'decoder_plugin': 'default',
        'window_size': 128,
        'threshold_error': 0.0003,
        'initial_size': 8,
        'step_size': 4,
        'remote_log': None,
        'remote_config': None,
        'load_config': './config_in.json',
        'save_config': './config_out.json',
        'quiet_mode': False,
        'force_date': False,
        'incremental_search': True,
        'headers': False,
        'epochs': 5,
        'batch_size': 256
    }
    print(f"Desired Step 1 Output: {desired_step1_output}")
    print(f"Step 1 - Default config: {merged_config}")
    if merged_config != desired_step1_output:
        print("Error: Step 1 output does not match the desired output.")
        sys.exit(1)

    # Step 2: Merge with plugin default parameters
    merged_config.update(encoder_plugin.plugin_params)
    merged_config.update(decoder_plugin.plugin_params)

    # Desired output for Step 2
    desired_step2_output = {
        'csv_file': './csv_input.csv',
        'save_encoder': './encoder_model.h5',
        'save_decoder': './decoder_model.h5',
        'load_encoder': None,
        'load_decoder': None,
        'evaluate_encoder': './encoder_eval.csv',
        'evaluate_decoder': './decoder_eval.csv',
        'encoder_plugin': 'default',
        'decoder_plugin': 'default',
        'window_size': 128,
        'threshold_error': 0.0003,
        'initial_size': 8,
        'step_size': 4,
        'remote_log': None,
        'remote_config': None,
        'load_config': './config_in.json',
        'save_config': './config_out.json',
        'quiet_mode': False,
        'force_date': False,
        'incremental_search': True,
        'headers': False,
        'epochs': 10,
        'batch_size': 256,
        'intermediate_layers': 1,
        'layer_size_divisor': 2
    }
    print(f"Desired Step 2 Output: {desired_step2_output}")
    print(f"Step 2 - Plugin defaults merged: {merged_config}")
    if merged_config != desired_step2_output:
        print("Error: Step 2 output does not match the desired output.")
        sys.exit(1)

    # Step 3: Merge with file configuration
    merged_config.update(config)

    # Desired output for Step 3
    desired_step3_output = {
        'csv_file': './csv_input.csv',
        'save_encoder': './encoder_model.h5',
        'save_decoder': './decoder_model.h5',
        'load_encoder': None,
        'load_decoder': None,
        'evaluate_encoder': './encoder_eval.csv',
        'evaluate_decoder': './decoder_eval.csv',
        'encoder_plugin': 'default',
        'decoder_plugin': 'default',
        'window_size': 128,
        'threshold_error': 0.0003,
        'initial_size': 8,
        'step_size': 4,
        'remote_log': None,
        'remote_config': None,
        'load_config': './config_in.json',
        'save_config': './config_out.json',
        'quiet_mode': False,
        'force_date': False,
        'incremental_search': True,
        'headers': False,
        'epochs': 5,
        'batch_size': 256,
        'intermediate_layers': 2,
        'layer_size_divisor': 2
    }
    print(f"Desired Step 3 Output: {desired_step3_output}")
    print(f"Step 3 - File config merged: {merged_config}")
    if merged_config != desired_step3_output:
        print("Error: Step 3 output does not match the desired output.")
        sys.exit(1)

    # Step 4: Merge with CLI arguments (ensure CLI args always override)
    cli_args_filtered = {k: v for k, v in cli_args.items() if v is not None}
    for key, value in cli_args_filtered.items():
        merged_config[key] = value

    # Desired output for Step 4
    desired_step4_output = {
        'csv_file': 'tests\\data\\csv_sel_unb_norm_512.csv',
        'save_encoder': './encoder_model.h5',
        'save_decoder': './decoder_model.h5',
        'load_encoder': None,
        'load_decoder': None,
        'evaluate_encoder': './encoder_eval.csv',
        'evaluate_decoder': './decoder_eval.csv',
        'encoder_plugin': 'lstm',
        'decoder_plugin': 'lstm',
        'window_size': 32,
        'threshold_error': 0.0003,
        'initial_size': 4,
        'step_size': 4,
        'remote_log': None,
        'remote_config': None,
        'load_config': 'input_config.json',
        'save_config': './config_out.json',
        'quiet_mode': False,
        'force_date': False,
        'incremental_search': False,
        'headers': False,
        'epochs': 5,
        'batch_size': 256,
        'intermediate_layers': 0,
        'layer_size_divisor': 2
    }
    print(f"Desired Step 4 Output: {desired_step4_output}")
    print(f"Step 4 - CLI arguments merged: {merged_config}")
    if merged_config != desired_step4_output:
        print("Error: Step 4 output does not match the desired output.")
        sys.exit(1)

    final_config = {k: v for k, v in merged_config.items() if k in config or k in cli_args_filtered or k in DEFAULT_VALUES}
    print(f"Final merged config: {final_config}")

    return final_config

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
