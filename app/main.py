import sys
import pandas as pd
import numpy as np
from config_handler import load_config, save_config, merge_config, save_debug_info, load_remote_config, save_remote_config, log_remote_data
from plugin_loader import load_encoder_decoder_plugins
from cli import parse_args
from data_handler import load_csv, write_csv

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
    if args.load_config:
        config = load_config(args.load_config)
        print(f"Loaded config from {args.load_config}: {config}")

    if args.remote_config:
        config = load_remote_config(args.remote_config, args.remote_username, args.remote_password)
        print(f"Loaded remote config from {args.remote_config}: {config}")

    print(f"Initial loaded config: {config}")
    print("Merging configuration with CLI arguments...")
    config = merge_config(config, cli_args, {})
    print(f"Config after merging with CLI args: {config}")

    print("Loading data...")
    data = load_csv(config['csv_file'])
    print(f"Loaded data from {config['csv_file']}")

    print("Loading encoder and decoder plugins...")
    encoder_plugin, encoder_params, decoder_plugin, decoder_params = load_encoder_decoder_plugins(config['encoder_plugin'], config['decoder_plugin'])

    encoder_plugin = encoder_plugin()
    decoder_plugin = decoder_plugin()

    print("Configuring encoder and decoder sizes...")
    encoder_plugin.configure_size(data.shape[1], config['initial_size'])
    decoder_plugin.configure_size(config['initial_size'], data.shape[1])

    current_size = config['initial_size']
    while True:
        print(f"Training encoder and decoder with interface size {current_size}...")
        encoder_plugin.train(data)
        encoded_data = encoder_plugin.encode(data)
        decoder_plugin.train(encoded_data, data)

        reconstructed_data = decoder_plugin.decode(encoded_data)
        mse = encoder_plugin.calculate_mse(data, reconstructed_data)
        print(f"Reconstruction MSE: {mse}")

        if mse <= config['max_error']:
            print(f"Reached acceptable error level: {mse} <= {config['max_error']}")
            break

        current_size -= config['step_size']
        if current_size <= 0:
            print("Minimum interface size reached, stopping training.")
            break

        print(f"Reducing interface size to {current_size}...")
        encoder_plugin.configure_size(data.shape[1], current_size)
        decoder_plugin.configure_size(current_size, data.shape[1])

    print(f"Saving encoder to {config['save_encoder']}...")
    encoder_plugin.save(config['save_encoder'])

    print(f"Saving decoder to {config['save_decoder']}...")
    decoder_plugin.save(config['save_decoder'])

    print(f"Saving final configuration to {config['save_config']}...")
    save_config(config, config['save_config'])

    print("Saving debug info...")
    encoder_debug_info = encoder_plugin.get_debug_info()
    decoder_debug_info = decoder_plugin.get_debug_info()
    save_debug_info({'encoder': encoder_debug_info, 'decoder': decoder_debug_info}, 'debug_out.json')

    print(f"Final reconstruction MSE: {mse}")

if __name__ == "__main__":
    main()
