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

    The final priority is:
         Highest: CLI arguments,
         Then: loaded parameters (file_config),
         Then: defaults (config.py),
         Then: plugin-specific parameters.
         
    This version expects six arguments.
    """
    # Step 1: Start with plugin-specific parameters (lowest priority)
    merged_config = {}
    for k, v in plugin_params1.items():
        print(f"Step 1 merging: plugin_param1 {k} = {v}")
        merged_config[k] = v
    for k, v in plugin_params2.items():
        print(f"Step 1.5 merging: plugin_param2 {k} = {v}")
        merged_config[k] = v
    print(f"After merging plugin-specific parameters: {merged_config}")
    
    # Step 2: Merge with default values from config.py (these override plugin-specific parameters)
    for k, v in defaults.items():
        print(f"Step 2 merging from defaults: {k} = {v}")
        merged_config[k] = v
    print(f"After merging defaults: {merged_config}")
    
    # Step 3: Merge with file configuration (these override defaults)
    for k, v in file_config.items():
        print(f"Step 3 merging from file config: {k} = {v}")
        merged_config[k] = v
    print(f"After merging file config: {merged_config}")
    
    # Step 4: Merge with CLI arguments (ensure CLI args always override)
    for k, v in cli_args.items():
        if v is not None:
            print(f"Step 4 merging from CLI args: {k} = {v}")
            merged_config[k] = v
    for key, value in process_unknown_args(unknown_args).items():
        converted_value = convert_type(value)
        print(f"Step 4 merging from unknown args: {key} = {converted_value}")
        merged_config[key] = converted_value

    # Extra check: if the CLI flag "--use_sliding_window" (singular) is provided and set to True,
    # then force "use_sliding_windows" to True.
    unknown = process_unknown_args(unknown_args)
    if ("use_sliding_window" in cli_args and str(cli_args["use_sliding_window"]).lower() == "true") or \
       ("use_sliding_window" in unknown and str(unknown["use_sliding_window"]).lower() == "true"):
        print("Overriding 'use_sliding_windows' to True based on CLI flag 'use_sliding_window'.")
        merged_config["use_sliding_windows"] = True

    # Special handling for positional csv_file
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        merged_config['x_train_file'] = sys.argv[1]
    
    print(f"Final merged configuration: {merged_config}")
    return merged_config
