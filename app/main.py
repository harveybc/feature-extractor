import sys
import json
from app.config_handler import load_config, save_config, merge_config
from app.cli import parse_args
from app.data_processor import process_data
from app.config import DEFAULT_VALUES
from app.autoencoder_manager import AutoencoderManager
from app.plugins.encoder_plugin_ann import Plugin as EncoderPlugin
from app.plugins.decoder_plugin_ann import Plugin as DecoderPlugin

def main():
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    print(f"Initial args: {args}")
    print(f"Unknown args: {unknown_args}")

    if unknown_args:
        print(f"Error: Unrecognized arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)

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

    print("Merging configuration with CLI arguments and unknown args...")
    config = merge_config(config, cli_args, {})
    print(f"Config after merging: {config}")

    if args.save_config:
        print(f"Saving configuration to {args.save_config}...")
        save_config(config, args.save_config)
        print(f"Configuration saved to {args.save_config}.")

    print("Processing data...")
    encoder_plugin = EncoderPlugin()
    decoder_plugin = DecoderPlugin()

    autoencoder_manager = AutoencoderManager(
        encoder_plugin, 
        decoder_plugin, 
        input_dim=config['window_size'], 
        encoding_dim=config['initial_encoding_dim']
    )

    autoencoder_manager.build_autoencoder()
    data = process_data(config)

    autoencoder_manager.train_autoencoder(
        data, 
        epochs=config['epochs'], 
        batch_size=config['batch_size']
    )

    encoded_data = autoencoder_manager.encode(data)
    decoded_data = autoencoder_manager.decode(encoded_data)

    if not args.quiet_mode:
        print("Encoded data:")
        print(encoded_data)
        print("Decoded data:")
        print(decoded_data)

if __name__ == "__main__":
    main()
