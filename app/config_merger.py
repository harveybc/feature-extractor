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

def merge_config(defaults, plugin_params1, plugin_params2, file_config, cli_args, unknown_args):
    """
    Merge configuration from multiple sources:
    1. 'defaults': A base dictionary of default values (e.g., DEFAULT_VALUES).
    2. 'plugin_params1': Dictionary of default parameters from the first plugin.
    3. 'plugin_params2': Dictionary of default parameters from the second plugin (optional usage).
    4. 'file_config': Configuration loaded from a file or remote source.
    5. 'cli_args': CLI arguments parsed by argparse (converted to a dict).
    6. 'unknown_args': Additional unknown arguments provided in the CLI.

    The merging order ensures that if a key exists in multiple dictionaries,
    the latter dictionary in the sequence overrides the earlier one.

    This version expects six arguments, unlike the original that only handled five.
    """

    # Step 1: Start with default values from config.py
    merged_config = defaults.copy()
    print(f"Actual Step 1 Output: {merged_config}")

    # Step 2: Merge with plugin_params1
    for k, v in plugin_params1.items():
        print(f"Step 2 merging: plugin_param1 {k} = {v}")
        merged_config[k] = v

    # Step 2.5: Merge with plugin_params2
    for k, v in plugin_params2.items():
        print(f"Step 2.5 merging: plugin_param2 {k} = {v}")
        merged_config[k] = v

    # Step 3: Merge with file configuration
    for k, v in file_config.items():
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
        merged_config['x_train_file'] = sys.argv[1]

    print(f"Actual Step 4 Output: {merged_config}")
    return merged_config

