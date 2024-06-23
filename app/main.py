import sys
import json
import pandas as pd
from app.config_handler import load_config, save_config, merge_config, save_debug_info
from app.cli import parse_args
from app.data_processor import process_data, run_autoencoder_pipeline
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin

def main():
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    print(f"Initial args: {args}")
    print(f"Unknown args: {unknown_args}")

    cli_args = vars(args)
    print(f"CLI arguments: {cli_args}")

    print("Loading default configuration...")
    config = DEFAULT_VALUES.copy()
    print(f"Default config: {config}")

    if args.load_config:
        file_config = load_config(args.load_config)
        print(f"Loaded config from file: {file_config}")
        config.update(file_config)
        print(f"Config after loading from file: {config}")

    print("Loading encoder plugin: ", config['encoder_plugin'])
    encoder_plugin_class, _ = load_plugin('feature_extractor.encoders', config['encoder_plugin'])
    print("Loading decoder plugin: ", config['decoder_plugin'])
    decoder_plugin_class, _ = load_plugin('feature_extractor.decoders', config['decoder_plugin'])

    encoder_plugin = encoder_plugin_class()
    decoder_plugin = decoder_plugin_class()

    print("Merging configuration with CLI arguments and unknown args...")
    config = merge_config(config, cli_args, dict(unknown_args), encoder_plugin, decoder_plugin)
    print(f"Config after merging: {config}")

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
