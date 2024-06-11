import sys
import json
import pandas as pd
from app.config_handler import load_config, save_config, merge_config, save_debug_info
from app.cli import parse_args
from app.data_processor import process_data
from app.config import DEFAULT_VALUES
from app.autoencoder_manager import AutoencoderManager
from app.data_handler import write_csv

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
    processed_data, debug_info = process_data(config)

    for column, windowed_data in processed_data.items():
        autoencoder_manager = AutoencoderManager(input_dim=windowed_data.shape[1], encoding_dim=config['initial_encoding_dim'])
        autoencoder_manager.build_autoencoder()
        autoencoder_manager.train_autoencoder(windowed_data, epochs=config['epochs'], batch_size=config['training_batch_size'])

        encoder_model_filename = f"{config['save_encoder_path']}_{column}.keras"
        decoder_model_filename = f"{config['save_decoder_path']}_{column}.keras"
        autoencoder_manager.save_encoder(encoder_model_filename)
        autoencoder_manager.save_decoder(decoder_model_filename)
        print(f"Saved encoder model to {encoder_model_filename}")
        print(f"Saved decoder model to {decoder_model_filename}")

        encoded_data = autoencoder_manager.encode_data(windowed_data)
        decoded_data = autoencoder_manager.decode_data(encoded_data)

        mse = autoencoder_manager.calculate_mse(windowed_data, decoded_data)
        print(f"Mean Squared Error for column {column}: {mse}")

        reconstructed_data = pd.DataFrame(decoded_data)
        output_filename = f"{config['csv_output_path']}_{column}.csv"
        write_csv(output_filename, reconstructed_data.values, include_date=config['force_date'], headers=config['headers'])
        print(f"Output written to {output_filename}")

        print(f"Encoder Dimensions: {autoencoder_manager.encoder_model.input_shape} -> {autoencoder_manager.encoder_model.output_shape}")
        print(f"Decoder Dimensions: {autoencoder_manager.decoder_model.input_shape} -> {autoencoder_manager.decoder_model.output_shape}")

if __name__ == "__main__":
    main()
