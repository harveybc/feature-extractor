# config_merger.py

def process_unknown_args(unknown_args):
    return {unknown_args[i].lstrip('--'): unknown_args[i + 1] for i in range(0, len(unknown_args), 2)}

def merge_config(default_config, encoder_plugin_params, decoder_plugin_params, file_config, cli_args, unknown_args):
    merged_config = default_config.copy()

    # Step 1: Start with default values
    print(f"Step 1 - Desired Output: {default_config}")
    print(f"Step 1 - Actual Output: {merged_config}")

    # Step 2: Merge with plugin default parameters
    for k, v in encoder_plugin_params.items():
        print(f"Step 2 - Merging from encoder plugin: {k} = {v}")
        merged_config[k] = v
    for k, v in decoder_plugin_params.items():
        print(f"Step 2 - Merging from decoder plugin: {k} = {v}")
        merged_config[k] = v
    print(f"Step 2 - Desired Output: {merged_config}")
    print(f"Step 2 - Actual Output: {merged_config}")

    # Step 3: Merge with file configuration
    for k, v in file_config.items():
        print(f"Step 3 - Merging from file config: {k} = {v}")
        merged_config[k] = v
    print(f"Step 3 - Desired Output: {merged_config}")
    print(f"Step 3 - Actual Output: {merged_config}")

    # Step 4: Merge with CLI arguments (ensure CLI args always override)
    cli_keys = [arg.lstrip('--') for arg in cli_args.keys()]
    for key, value in cli_args.items():
        if key in cli_keys and value is not None:
            print(f"Step 4 - Merging from CLI args: {key} = {value}")
            merged_config[key] = value

    unknown_keys = [arg.lstrip('--') for arg in unknown_args.keys()]
    for key, value in unknown_args.items():
        if key in unknown_keys and value is not None:
            print(f"Step 4 - Merging from unknown args: {key} = {value}")
            merged_config[key] = value

    print(f"Step 4 - Desired Output: {merged_config}")
    print(f"Step 4 - Actual Output: {merged_config}")

    final_config = {}
    for k, v in merged_config.items():
        if k in default_config or k in cli_args or k in file_config or k in encoder_plugin_params or k in decoder_plugin_params:
            print(f"Final merging: {k} = {v}")
            final_config[k] = v

    return final_config
