import sys
import json
import pandas as pd
from app.config_handler import load_config, save_config
from app.cli import parse_args
from app.data_processor import process_data, run_autoencoder_pipeline
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

def main():
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()

    cli_args = vars(args)

    print("Loading default configuration...")
    config = DEFAULT_VALUES.copy()

    file_config = {}
    if args.load_config:
        file_config = load_config(args.load_config)
        print(f"Loaded config from file: {file_config}")
        config.update(file_config)

    encoder_plugin_name = cli_args['encoder_plugin']
    decoder_plugin_name = cli_args['decoder_plugin']

    print(f"Loading encoder plugin: {encoder_plugin_name}")
    encoder_plugin_class, _ = load_plugin('feature_extractor.encoders', encoder_plugin_name)
    print(f"Loading decoder plugin: {decoder_plugin_name}")
    decoder_plugin_class, _ = load_plugin('feature_extractor.decoders', decoder_plugin_name)

    encoder_plugin = encoder_plugin_class()
    decoder_plugin = decoder_plugin_class()

    print("Merging configuration with CLI arguments and unknown args...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, encoder_plugin.plugin_params, decoder_plugin.plugin_params, file_config, cli_args, unknown_args_dict)

    if args.save_config:
        print(f"Saving configuration to {args.save_config}...")
        save_config(config, args.save_config)
        print(f"Configuration saved to {args.save_config}.")

    encoder_plugin.set_params(**config)
    decoder_plugin.set_params(**config)

    print("Processing and running autoencoder pipeline...")
    run_autoencoder_pipeline(config, encoder_plugin, decoder_plugin)

if __name__ == "__main__":
    main()
