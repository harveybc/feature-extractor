# config_merger.py

import sys
from app.config import DEFAULT_VALUES

def process_unknown_args(unknown_args):
    return {unknown_args[i].lstrip('--'): unknown_args[i + 1] for i in range(0, len(unknown_args), 2)}

def convert_type(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def merge_config(defaults, encoder_plugin_params, decoder_plugin_params, config, cli_args, unknown_args):
    
    # Step 1: Start with default values from config.py
    merged_config = defaults.copy()
    
    print(f"Actual Step 1 Output: {merged_config}")
    
    # Step 2: Merge with plugin default parameters
    for k, v in encoder_plugin_params.items():
        print(f"Step 2 merging: encoder_plugin_param {k} = {v}")
        merged_config[k] = v
    for k, v in decoder_plugin_params.items():
        print(f"Step 2 merging: decoder_plugin_param {k} = {v}")
        merged_config[k] = v
    
    print(f"Actual Step 2 Output: {merged_config}")
    
    # Step 3: Merge with file configuration
    for k, v in config.items():
        print(f"Step 3 merging from file config: {k} = {v}")
        merged_config[k] = v
    
    print(f"Actual Step 3 Output: {merged_config}")

    # Step 4: Merge with CLI arguments (ensure CLI args always override)
    cli_keys = [arg.lstrip('--') for arg in sys.argv if arg.startswith('--')]
    for key in cli_keys:
        if key in cli_args:
            print(f"Step 4 merging from CLI args: {key} = {cli_args[key]}")
            merged_config[key] = cli_args[key]
        elif key in unknown_args:
            value = convert_type(unknown_args[key])
            print(f"Step 4 merging from unknown args: {key} = {value}")
            merged_config[key] = value
    
    # Special handling for csv_file
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        merged_config['input_file'] = sys.argv[1]
    
    print(f"Actual Step 4 Output: {merged_config}")
    
    return merged_config
