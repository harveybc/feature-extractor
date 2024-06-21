import sys
import json
import pandas as pd
from app.config_handler import load_config, save_config, merge_config, save_debug_info
from app.cli import parse_args
from app.data_processor import process_data, train_autoencoder
from app.config import DEFAULT_VALUES
from app.data_handler import write_csv
from app.plugin_loader import load_encoder_decoder_plugins

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

    print("Loading encoder plugin: ", config['encoder_plugin'])
    encoder_plugin_class, encoder_params = load_plugin('feature_extractor.encoders', config['encoder_plugin'])
    print("Loading decoder plugin: ", config['decoder_plugin'])
    decoder_plugin_class, decoder_params = load_plugin('feature_extractor.decoders', config['decoder_plugin'])

    encoder_plugin = encoder_plugin_class()
    decoder_plugin = decoder_plugin_class()

    print("Processing data...")
    processed_data, debug_info = process_data(config)

    for column, windowed_data in processed_data.items():
        encoder_plugin.configure_size(input_dim=windowed_data.shape[1], encoding_dim=config['initial_encoding_dim'])
        decoder_plugin.configure_size(encoding_dim=config['initial_encoding_dim'], output_dim=windowed_data.shape[1])

        autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
        trained_manager = train_autoencoder(autoencoder_manager, windowed_data, config['mse_threshold'], config['initial_encoding_dim'], config['encoding_step_size'], config['incremental_search'], config['epochs'])

        encoder_model_filename = f"{config['save_encoder_path']}_{column}.keras"
        decoder_model_filename = f"{config['save_decoder_path']}_{column}.keras"
        trained_manager.save_encoder(encoder_model_filename)
        trained_manager.save_decoder(decoder_model_filename)
        print(f"Saved encoder model to {encoder_model_filename}")
        print(f"Saved decoder model to {decoder_model_filename}")

        encoded_data = trained_manager.encode_data(windowed_data)
        decoded_data = trained_manager.decode_data(encoded_data)

        mse = trained_manager.calculate_mse(windowed_data, decoded_data)
        print(f"Mean Squared Error for column {column}: {mse}")

        reconstructed_data = pd.DataFrame(decoded_data)
        output_filename = f"{config['csv_output_path']}_{column}.csv"
        write_csv(output_filename, reconstructed_data.values, include_date=config['force_date'], headers=config['headers'])
        print(f"Output written to {output_filename}")

        print(f"Encoder Dimensions: {trained_manager.encoder_model.input_shape} -> {trained_manager.encoder_model.output_shape}")
        print(f"Decoder Dimensions: {trained_manager.decoder_model.input_shape} -> {trained_manager.decoder_model.output_shape}")

if __name__ == "__main__":
    main()