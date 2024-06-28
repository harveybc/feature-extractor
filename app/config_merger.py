# config_merger.py

import sys
from app.config import DEFAULT_VALUES

def merge_config(defaults, config, cli_args, unknown_args, encoder_plugin, decoder_plugin):
    def exit_on_error(step, actual, desired):
        if actual != desired:
            print(f"Error: {step} output does not match the desired output.")
            print(f"Desired Output: {desired}")
            print(f"Actual Output: {actual}")
            sys.exit(1)

    # Step 1: Start with default values from config.py
    merged_config = defaults.copy()
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
        'threshold_error': 0.003,
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
        'intermediate_layers': 2
    }
    exit_on_error("Step 1", merged_config, desired_step1_output)

    # Step 2: Merge with plugin default parameters
    for k, v in encoder_plugin.plugin_params.items():
        merged_config[k] = v
    for k, v in decoder_plugin.plugin_params.items():
        merged_config[k] = v
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
        'threshold_error': 0.003,
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
    exit_on_error("Step 2", merged_config, desired_step2_output)

    # Step 3: Merge with file configuration
    for k, v in config.items():
        merged_config[k] = v
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
        'threshold_error': 0.003,
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
        'intermediate_layers': 2,
        'layer_size_divisor': 2
    }
    exit_on_error("Step 3", merged_config, desired_step3_output)

    # Step 4: Merge with CLI arguments (ensure CLI args always override)
    cli_keys = [arg.lstrip('--') for arg in sys.argv if arg.startswith('--')]
    for key in cli_keys:
        if key in cli_args:
            merged_config[key] = cli_args[key]
    desired_step4_output = {
        'csv_file': 'tests\\data\\csv_sel_unb_norm_512.csv',
        'save_encoder': './encoder_model.h5',
        'save_decoder': './decoder_model.h5',
        'load_encoder': None,
        'load_decoder': None,
        'evaluate_encoder': './encoder_eval.csv',
        'evaluate_decoder': './decoder_eval.csv',
        'encoder_plugin': 'cnn',
        'decoder_plugin': 'cnn',
        'window_size': 32,
        'threshold_error': 0.003,
        'initial_size': 4,
        'step_size': 4,
        'remote_log': None,
        'remote_config': None,
        'load_config': 'input_config.json',
        'save_config': './config_out.json',
        'quiet_mode': False,
        'force_date': False,
        'incremental_search': True,
        'headers': False,
        'epochs': 10,
        'batch_size': 256,
        'intermediate_layers': 4,
        'layer_size_divisor': 2
    }
    exit_on_error("Step 4", merged_config, desired_step4_output)

    return merged_config
