import sys
import requests
import numpy as np
import pandas as pd
from app.data_handler import load_csv, write_csv, sliding_window
from app.plugin_loader import load_encoder_decoder_plugins

def train_autoencoder(encoder, decoder, data, max_error, initial_size, step_size):
    current_size = initial_size
    current_mse = float('inf')

    while current_size > 0 and current_mse > max_error:
        print(f"Configuring encoder and decoder with interface size {current_size}...")
        encoder.configure_size(input_dim=data.shape[1], encoding_dim=current_size)
        decoder.configure_size(encoding_dim=current_size, output_dim=data.shape[1])

        print(f"Training encoder with interface size {current_size}...")
        encoder.train(data)

        encoded_data = encoder.encode(data)

        print(f"Training decoder with interface size {current_size}...")
        decoder.train(encoded_data, data)

        decoded_data = decoder.decode(encoded_data)
        current_mse = encoder.calculate_mse(data, decoded_data)
        print(f"Current MSE: {current_mse} at interface size: {current_size}")

        if current_mse <= max_error:
            print("Desired MSE reached. Stopping training.")
            break

        current_size -= step_size

    return encoder, decoder

def process_data(config):
    print(f"Loading data from {config['csv_file']}...")
    data = load_csv(config['csv_file'], headers=config['headers'])
    print(f"Data loaded: {len(data)} rows and {data.shape[1]} columns.")
    
    # Ensure all data is numerical
    print("Converting data to float...")
    data = data.apply(pd.to_numeric, errors='coerce')
    print(f"Data types:\n{data.dtypes}")

    debug_info = {}

    for col in data.columns:
        column_data = data[[col]]
        print(f"Processing column: {col}")
        windowed_data = sliding_window(column_data, config['window_size'])
        print(f"Data loaded and windowed for column {col}. Number of windows: {len(windowed_data)}.")

        encoder_name = config.get('encoder_plugin', 'default')
        decoder_name = config.get('decoder_plugin', 'default')
        Encoder, encoder_params, Decoder, decoder_params = load_encoder_decoder_plugins(encoder_name, decoder_name)

        encoder = Encoder()
        decoder = Decoder()

        for index, series_data in enumerate(windowed_data):
            print(f"Training autoencoder for window {index} on column {col}...")
            
            # Ensure all data is numerical for each window
            series_data = np.asarray(series_data).astype(np.float32)
            print(f"Window data types for column {col}:\n{series_data.dtype}")

            trained_encoder, trained_decoder = train_autoencoder(encoder, decoder, series_data, config['max_error'], config['initial_size'], config['step_size'])

            if col == data.columns[0]:
                encoder_filename = config['save_encoder']
                decoder_filename = config['save_decoder']
            else:
                encoder_filename = f"{config['save_encoder']}_{col}"
                decoder_filename = f"{config['save_decoder']}_{col}"

            print(f"Saving encoder model to {encoder_filename}...")
            trained_encoder.save(encoder_filename)

            print(f"Saving decoder model to {decoder_filename}...")
            trained_decoder.save(decoder_filename)

            encoded_data = trained_encoder.encode(series_data)
            decoded_data = trained_decoder.decode(encoded_data)

            mse = encoder.calculate_mse(series_data, decoded_data)
            print(f"Mean Squared Error for window {index} on column {col}: {mse}")
            debug_info[f'mean_squared_error_{index}_{col}'] = mse

            trained_encoder.add_debug_info(debug_info)
            trained_decoder.add_debug_info(debug_info)

            output_filename = f"{config['csv_output_path']}_{index}_{col}.csv"
            print(f"Writing decoded data to {output_filename}...")
            write_csv(output_filename, decoded_data, include_date=config['force_date'], headers=config['headers'])

            if config['remote_log']:
                log_response = requests.post(
                    config['remote_log'],
                    auth=(config['remote_username'], config['remote_password']),
                    json=debug_info
                )
                print(f"Remote log response: {log_response.text}")

    return decoded_data, debug_info

