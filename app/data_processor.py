import sys
import requests
import numpy as np
import pandas as pd
from app.data_handler import load_csv, write_csv, sliding_window
from app.plugin_loader import load_encoder_decoder_plugins
from app.reconstruction import unwindow_data
from app.autoencoder_manager import AutoencoderManager
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_autoencoder(autoencoder_manager, data, mse_threshold, initial_size, step_size, incremental_search, epochs):
    current_size = initial_size
    current_mse = float('inf')
    print(f"Training autoencoder with initial size {current_size}...")

    while current_size > 0 and ((current_mse > mse_threshold) if not incremental_search else (current_mse < mse_threshold)):
        autoencoder_manager.build_autoencoder()
        autoencoder_manager.train_autoencoder(data, epochs=epochs, batch_size=256)

        encoded_data = autoencoder_manager.encode_data(data)
        decoded_data = autoencoder_manager.decode_data(encoded_data)
        current_mse = autoencoder_manager.calculate_mse(data, decoded_data)
        print(f"Current MSE: {current_mse} at interface size: {current_size}")

        if (incremental_search and current_mse >= mse_threshold) or (not incremental_search and current_mse <= mse_threshold):
            print("Desired MSE reached. Stopping training.")
            break

        if incremental_search:
            current_size += step_size
            if current_size >= data.shape[1]:
                break
        else:
            current_size -= step_size

    return autoencoder_manager

def process_data(config):
    data = load_csv(config['csv_file'], headers=config['headers'])
    print(f"Data loaded: {data.shape[0]} rows and {data.shape[1]} columns.")

    if config['force_date']:
        data.index = pd.to_datetime(data.index)

    print(f"Data types:\n {data.dtypes}")

    debug_info = {}

    for column in data.columns:
        print(f"Processing column: {column}")
        column_data = data[[column]].values.astype(np.float64)
        windowed_data = sliding_window(column_data, config['window_size'])
        windowed_data = windowed_data.squeeze()  # Ensure correct shape for training
        print(f"Windowed data shape: {windowed_data.shape}")

        autoencoder_manager = AutoencoderManager(input_dim=windowed_data.shape[1], encoding_dim=config['initial_encoding_dim'])
        trained_autoencoder_manager = train_autoencoder(
            autoencoder_manager, windowed_data, config['mse_threshold'], 
            config['initial_encoding_dim'], config['encoding_step_size'], 
            config['incremental_search'], config['epochs']
        )

        encoder_model_filename = f"{config['save_encoder_path']}_{column}.keras"
        decoder_model_filename = f"{config['save_decoder_path']}_{column}.keras"
        trained_autoencoder_manager.save_encoder(encoder_model_filename)
        trained_autoencoder_manager.save_decoder(decoder_model_filename)
        print(f"Saved encoder model to {encoder_model_filename}")
        print(f"Saved decoder model to {decoder_model_filename}")

        encoded_data = trained_autoencoder_manager.encode_data(windowed_data)
        decoded_data = trained_autoencoder_manager.decode_data(encoded_data)

        mse = mean_squared_error(windowed_data, decoded_data)
        mae = mean_absolute_error(windowed_data, decoded_data)
        print(f"Mean Squared Error for column {column}: {mse}")
        print(f"Mean Absolute Error for column {column}: {mae}")
        debug_info[f'mean_squared_error_{column}'] = mse
        debug_info[f'mean_absolute_error_{column}'] = mae

        reconstructed_data = unwindow_data(pd.DataFrame(decoded_data))

        output_filename = f"{config['csv_output_path']}_{column}.csv"
        write_csv(output_filename, reconstructed_data.values, include_date=config['force_date'], headers=config['headers'])
        print(f"Output written to {output_filename}")

        print(f"Encoder Dimensions: {trained_autoencoder_manager.encoder_model.input_shape} -> {trained_autoencoder_manager.encoder_model.output_shape}")
        print(f"Decoder Dimensions: {trained_autoencoder_manager.decoder_model.input_shape} -> {trained_autoencoder_manager.decoder_model.output_shape}")

    return reconstructed_data, debug_info
