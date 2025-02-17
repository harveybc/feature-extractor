# config_merger.py

import sys
from app.config import DEFAULT_VALUES

def process_unknown_args(unknown_args):
    """
    Process a list of unknown CLI arguments into a dictionary.
    This version expects that flags are provided in pairs (flag value),
    and will convert the value using convert_type.
    """
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
    3. 'plugin_params2': Dictionary of default parameters from the second plugin.
    4. 'file_config': Configuration loaded from a file or remote source.
    5. 'cli_args': CLI arguments parsed by argparse (converted to a dict).
    6. 'unknown_args': Additional unknown arguments provided in the CLI.
    
    The merging order ensures that if a key exists in multiple dictionaries,
    the following priority is used (from highest to lowest):
      (1) CLI arguments,
      (2) loaded parameters (file_config),
      (3) default values (defaults),
      (4) plugin-specific parameters.
      
    This version expects six arguments.
    """
    # Step 1: Start with plugin-specific parameters (lowest priority)
    merged_config = {}
    merged_config.update(plugin_params1)
    merged_config.update(plugin_params2)
    print(f"Step 1 (plugin-specific): {merged_config}")

    # Step 2: Merge in default values from config.py (overriding plugin-specific if needed)
    merged_config.update(defaults)
    print(f"Step 2 (after defaults): {merged_config}")

    # Step 3: Merge in loaded parameters (file/remote config) which override defaults and plugin params
    merged_config.update(file_config)
    print(f"Step 3 (after file_config): {merged_config}")

    # Step 4: Finally, merge CLI arguments and unknown args to override all previous values
    merged_config.update(cli_args)
    for key, value in process_unknown_args(list(unknown_args)).items():
        merged_config[key] = convert_type(value)
    print(f"Step 4 (after CLI and unknown): {merged_config}")

    # Special handling for positional csv_file (if provided)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        merged_config['x_train_file'] = sys.argv[1]

    print(f"Final merged configuration: {merged_config}")
    return merged_config
