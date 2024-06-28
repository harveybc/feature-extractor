import sys
from app.config import DEFAULT_VALUES

def merge_config(default_values, plugin_defaults, file_config, cli_args):
    # Step 1: Start with default values
    merged_config = default_values.copy()

    # Step 2: Merge with plugin default parameters
    for k, v in plugin_defaults.items():
        merged_config[k] = v

    # Step 3: Merge with file configuration
    for k, v in file_config.items():
        print(f"Merging from file config: {k} = {v}")
        merged_config[k] = v
    print(f"Step 3 - File config merged: {merged_config}")
    desired_step3_output = {'csv_file': './csv_input.csv', 'save_encoder': './encoder_model.h5', 'save_decoder': './decoder_model.h5', 'load_encoder': None, 'load_decoder': None, 'evaluate_encoder': './encoder_eval.csv', 'evaluate_decoder': './decoder_eval.csv', 'encoder_plugin': 'default', 'decoder_plugin': 'default', 'window_size': 128, 'threshold_error': 0.003, 'initial_size': 8, 'step_size': 4, 'remote_log': None, 'remote_config': None, 'load_config': './config_in.json', 'save_config': './config_out.json', 'quiet_mode': False, 'force_date': False, 'incremental_search': True, 'headers': False, 'epochs': 5, 'batch_size': 256, 'intermediate_layers': 2, 'layer_size_divisor': 2}
    print(f"Desired Step 3 Output: {desired_step3_output}")
    if merged_config != desired_step3_output:
        print(f"Actual Step 3 Output: {merged_config}")
        print("Error: Step 3 output does not match the desired output.")
        sys.exit(1)

    # Step 4: Merge with CLI arguments (ensure CLI args always override)
    user_set_cli_args = {}
    for arg in sys.argv:
        if arg.startswith("--"):
            key = arg[2:].split("=")[0]
            user_set_cli_args[key] = cli_args.get(key)

    for key, value in user_set_cli_args.items():
        if value is not None:
            print(f"Merging from CLI args: {key} = {value}")
            merged_config[key] = value
    print(f"Step 4 - CLI arguments merged: {merged_config}")
    desired_step4_output = {'csv_file': 'tests\\data\\csv_sel_unb_norm_512.csv', 'save_encoder': './encoder_model.h5', 'save_decoder': './decoder_model.h5', 'load_encoder': None, 'load_decoder': None, 'evaluate_encoder': './encoder_eval.csv', 'evaluate_decoder': './decoder_eval.csv', 'encoder_plugin': 'cnn', 'decoder_plugin': 'cnn', 'window_size': 32, 'threshold_error': 0.003, 'initial_size': 4, 'step_size': 4, 'remote_log': None, 'remote_config': None, 'load_config': 'input_config.json', 'save_config': './config_out.json', 'quiet_mode': False, 'force_date': False, 'incremental_search': True, 'headers': False, 'epochs': 5, 'batch_size': 256, 'intermediate_layers': 4, 'layer_size_divisor': 2}
    print(f"Desired Step 4 Output: {desired_step4_output}")
    print(f"Actual Step 4 Output: {merged_config}")
    if merged_config != desired_step4_output:
        print("Error: Step 4 output does not match the desired output.")
        sys.exit(1)

    final_config = {}
    for k, v in merged_config.items():
        if k in file_config or k in cli_args or k in default_values:
            final_config[k] = v

    return final_config
