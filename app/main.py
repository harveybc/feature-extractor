import sys
from app.cli import parse_args
from app.config_handler import load_config, save_config, merge_config, save_debug_info
from app.data_handler import load_csv, write_csv
from app.plugin_loader import load_encoder_decoder_plugins
from app.data_processor import process_data

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
    elif args.remote_config:
        config = load_remote_config(args.remote_config, args.remote_username, args.remote_password)

    print(f"Initial loaded config: {config}")

    print("Merging configuration with CLI arguments...")
    config = merge_config(config, cli_args, {})
    print(f"Config after merging with CLI args: {config}")

    print("Processing data...")
    decoded_data, debug_info = process_data(config)

    print("Saving final configuration...")
    save_config(config, config['save_config'])

    print("Saving debug information...")
    save_debug_info(debug_info)

if __name__ == "__main__":
    main()
