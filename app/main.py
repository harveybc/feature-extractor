import sys
import json
from app.config_handler import load_config, save_config, merge_config, save_debug_info
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
    data = process_data(config)

    # Initialize encoder and decoder plugins
    encoder_plugin = EncoderPlugin()
    decoder_plugin = DecoderPlugin()

    # Initialize autoencoder manager
    autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin, input_dim=128, encoding_dim=4)
    autoencoder_manager.build_autoencoder()

    # Train autoencoder
    autoencoder_manager.train_autoencoder(data, epochs=config['epochs'], batch_size=config['training_batch_size'])

    # Encode and decode data
    encoded_data = autoencoder_manager.encode(data)
    decoded_data = autoencoder_manager.decode(encoded_data)

    # Calculate MSE
    mse = autoencoder_manager.calculate_mse(data, decoded_data)
    print(f"Mean Squared Error: {mse}")

    # Save models if specified
    if config['save_encoder']:
        autoencoder_manager.save(config['save_encoder'], config['save_decoder'])

    if not args.quiet_mode:
        print("Processed data:")
        print(decoded_data)
        debug_info = {"mse": mse}
        print("Debug information:")
        print(json.dumps(debug_info, indent=4))

if __name__ == "__main__":
    main()
