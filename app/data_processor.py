import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from app.autoencoder_manager import AutoencoderManager
from app.data_handler import load_csv, write_csv
from app.reconstruction import unwindow_data
from app.config_handler import save_config, save_debug_info

def create_sliding_windows(data, window_size):
    data_array = data.to_numpy()
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data_array,
        targets=None,
        sequence_length=window_size,
        sequence_stride=1,
        batch_size=1
    )

    windows = []
    for batch in dataset:
        windows.append(batch.numpy().flatten())

    return pd.DataFrame(windows)

def process_data(config):
    print(f"Loading data from CSV file: {config['csv_file']}")
    data = load_csv(config['csv_file'], headers=config['headers'])
    print(f"Data loaded with shape: {data.shape}")

    window_size = config['window_size']
    print(f"Applying sliding window of size: {window_size}")
    windowed_data = create_sliding_windows(data, window_size)
    print(f"Windowed data shape: {windowed_data.shape}")

    processed_data = {col: windowed_data.values for col in data.columns}
    debug_info = {'window_size': window_size, 'num_columns': len(windowed_data.columns)}
    print(f"Processed data: {list(processed_data.keys())}")
    print(f"Debug info: {debug_info}")

    return processed_data, debug_info

def run_autoencoder_pipeline(config, encoder_plugin, decoder_plugin):
    start_time = time.time()
    
    print("Running process_data...")
    processed_data, debug_info = process_data(config)
    print("Processed data received.")

    for column, windowed_data in processed_data.items():
        print(f"Processing column: {column}")
        autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
        
        window_size = config['window_size']
        interface_size = config['initial_size']
        
        autoencoder_manager.build_autoencoder(window_size, interface_size)
        
        initial_size = config['initial_size']
        step_size = config['step_size']
        threshold_error = config['threshold_error']
        training_batch_size = config['batch_size']
        epochs = config['epochs']
        
        current_size = initial_size
        while True:
            print(f"Training with interface size: {current_size}")
            encoder_plugin.configure_size(window_size, current_size)
            decoder_plugin.configure_size(current_size, window_size)

            autoencoder_manager.train_autoencoder(windowed_data, epochs=epochs, batch_size=training_batch_size)

            encoded_data = autoencoder_manager.encode_data(windowed_data)
            decoded_data = autoencoder_manager.decode_data(encoded_data)

            mse = autoencoder_manager.calculate_mse(windowed_data, decoded_data)
            mae = autoencoder_manager.calculate_mae(windowed_data, decoded_data)
            print(f"Mean Squared Error for column {column} with interface size {current_size}: {mse}")
            print(f"Mean Absolute Error for column {column} with interface size {current_size}: {mae}")

            if mse <= threshold_error:
                print(f"Optimal interface size found: {current_size} with MSE: {mse} and MAE: {mae}")
                break
            else:
                current_size += step_size
                if current_size > windowed_data.shape[1]:
                    print(f"Cannot increase interface size beyond data dimensions. Stopping.")
                    break

        encoder_model_filename = f"{config['save_encoder']}_{column}.keras"
        decoder_model_filename = f"{config['save_decoder']}_{column}.keras"
        autoencoder_manager.save_encoder(encoder_model_filename)
        autoencoder_manager.save_decoder(decoder_model_filename)
        print(f"Saved encoder model to {encoder_model_filename}")
        print(f"Saved decoder model to {decoder_model_filename}")

        # Perform unwindowing of the decoded data once
        reconstructed_data = unwindow_data(pd.DataFrame(decoded_data))

        output_filename = os.path.splitext(config['csv_file'])[0] + f"_{column}.csv"
        write_csv(output_filename, reconstructed_data, include_date=config['force_date'], headers=config['headers'], window_size=config['window_size'])
        print(f"Output written to {output_filename}")

        print(f"Encoder Dimensions: {autoencoder_manager.encoder_model.input_shape} -> {autoencoder_manager.encoder_model.output_shape}")
        print(f"Decoder Dimensions: {autoencoder_manager.decoder_model.input_shape} -> {autoencoder_manager.decoder_model.output_shape}")

    # Save final configuration and debug information
    end_time = time.time()
    execution_time = end_time - start_time
    debug_info['execution_time'] = execution_time

    save_config(config, config['save_config'])
    save_debug_info(debug_info, 'debug_out.json')

    print(f"Execution time: {execution_time} seconds")
