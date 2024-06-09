import sys
import json
import time
import requests
import numpy as np
from app.plugin_loader import load_plugin, load_encoder_decoder_plugins
from app.cli import parse_args
from app.config_handler import load_config, save_config, save_debug_info, merge_config, load_remote_config, save_remote_config, log_remote_data
from app.data_handler import load_csv, write_csv
from app.data_processor import process_data
import config

def main():
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    print(f"Initial args: {args}")
    print(f"Unknown args: {unknown_args}")

    cli_args = vars(args)
    print(f"CLI arguments: {cli_args}")

    # Convert unknown args to a dictionary
    unknown_args_dict = {}
    current_key = None
    for arg in unknown_args:
        if arg.startswith('--'):
            if current_key:
                unknown_args_dict[current_key] = True  # Flags without values are treated as True
            current_key = arg[2:]
        else:
            if current_key:
                unknown_args_dict[current_key] = arg
                current_key = None

    if current_key:
        unknown_args_dict[current_key] = True

    print(f"Unknown args as dict: {unknown_args_dict}")

    # Specific handling for --range argument
    if 'range' in unknown_args_dict:
        range_str = unknown_args_dict['range']
        try:
            unknown_args_dict['range'] = tuple(map(int, range_str.strip("()").split(',')))
        except ValueError:
            print(f"Error: Invalid format for --range argument: {range_str}", file=sys.stderr)
            return

    print("Loading configuration...")
    config = {}
    if args.remote_config:
        config = load_remote_config(args.remote_config, args.remote_username, args.remote_password)
    elif args.load_config:
        config = load_config(args.load_config)
    
    print(f"Initial loaded config: {config}")

    print("Merging configuration with CLI arguments...")
    config = merge_config(config, cli_args, unknown_args_dict)
    print(f"Config after merging with CLI args: {config}")

    debug_info = {
        "execution_time": "",
        "input_rows": 0,
        "output_rows": 0,
        "input_columns": 0,
        "output_columns": 0
    }

    start_time = time.time()

    if not config.get('csv_file'):
        print("Error: No CSV file specified.", file=sys.stderr)
        return

    data = load_csv(config['csv_file'], headers=config['headers'])
    debug_info["input_rows"] = len(data)
    debug_info["input_columns"] = len(data.columns)

    encoder_plugin, encoder_params, decoder_plugin, decoder_params = load_encoder_decoder_plugins(config['encoder_plugin'], config['decoder_plugin'])
    # Set plugin params (add actual implementation as needed)
    encoder_plugin.set_params(**{k: config[k] for k in encoder_params if k in config})
    decoder_plugin.set_params(**{k: config[k] for k in decoder_params if k in config})

    # Process data with encoder and decoder plugins
    decoded_data, debug_info = process_data(config, encoder_plugin, decoder_plugin)

    debug_info["output_rows"] = len(decoded_data)
    debug_info["output_columns"] = len(decoded_data[0].columns)

    if args.save_encoder:
        encoder_plugin.save(args.save_encoder)

    if args.save_decoder:
        decoder_plugin.save(args.save_decoder)

    if args.save_config:
        config_str, config_filename = save_config(config, args.save_config)
        print(f"Configuration saved to {config_filename}")

    execution_time = time.time() - start_time
    debug_info["execution_time"] = execution_time

    if 'debug_file' not in config or not config['debug_file']:
        config['debug_file'] = args.debug_file

    save_debug_info(debug_info, config['debug_file'])
    print(f"Debug info saved to {config['debug_file']}")
    print(f"Execution time: {execution_time} seconds")

    if args.remote_save_config:
        if save_remote_config(config, args.remote_save_config, args.remote_username, args.remote_password):
            print(f"Configuration successfully saved to remote URL {args.remote_save_config}")
        else:
            print(f"Failed to save configuration to remote URL {args.remote_save_config}")

    if args.remote_log:
        if log_remote_data(debug_info, args.remote_log, args.remote_username, args.remote_password):
            print(f"Debug information successfully logged to remote URL {args.remote_log}")
        else:
            print(f"Failed to log debug information to remote URL {args.remote_log}")

if __name__ == '__main__':
    main()
