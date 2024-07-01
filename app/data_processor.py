import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from app.autoencoder_manager import AutoencoderManager
from app.data_handler import load_csv, write_csv
from app.reconstruction import unwindow_data
from app.config_handler import save_debug_info, remote_log

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
    print(f"Loading data from CSV file: {config['input_file']}")
    data = load_csv(config['input_file'], headers=config['headers'])
    print(f"Data loaded with shape: {data.shape}")

    window_size = config['window_size']
    print(f"Applying sliding window of size: {window_size}")
    windowed_data = create_sliding_windows(data, window_size)
    print(f"Windowed data shape: {windowed_data.shape}")

    processed_data = {col: windowed_data.values for col in data.columns}
    return processed_data

def run_autoencoder_pipeline(config, encoder_plugin, decoder_plugin):
    start_time = time.time()
    
    print("Running process_data...")
    processed_data = process_data(config)
    print("Processed data received.")
    mse=0
    mae=0
    for column, windowed_data in processed_data.items():
        print(f"Processing column: {column}")
        
        # Training loop to optimize the latent space size
        initial_size = config['initial_size']
        step_size = config['step_size']
        threshold_error = config['threshold_error']
        training_batch_size = config['batch_size']
        epochs = config['epochs']
        incremental_search = config['incremental_search']
        
        current_size = initial_size
        while True:
            print(f"Training with interface size: {current_size}")
            
            # Create a new instance of AutoencoderManager for each iteration
            autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
            
            # Build new autoencoder model with the current size
            autoencoder_manager.build_autoencoder(config['window_size'], current_size)

            # Train the autoencoder model
            autoencoder_manager.train_autoencoder(windowed_data, epochs=epochs, batch_size=training_batch_size)

            # Encode and decode the data
            encoded_data = autoencoder_manager.encode_data(windowed_data)
            decoded_data = autoencoder_manager.decode_data(encoded_data)

            # Check if the decoded data needs reshaping
            if len(decoded_data.shape) == 3:
                decoded_data = decoded_data.reshape(decoded_data.shape[0], decoded_data.shape[1])

            # Calculate the MSE and MAE
            mse = autoencoder_manager.calculate_mse(windowed_data, decoded_data)
            mae = autoencoder_manager.calculate_mae(windowed_data, decoded_data)
            print(f"Mean Squared Error for column {column} with interface size {current_size}: {mse}")
            print(f"Mean Absolute Error for column {column} with interface size {current_size}: {mae}")

            if (incremental_search and mse <= threshold_error) or (not incremental_search and mse >= threshold_error):
                print(f"Optimal interface size found: {current_size} with MSE: {mse} and MAE: {mae}")
                break
            else:
                if incremental_search:
                    current_size += step_size
                else:
                    current_size -= step_size
                if current_size > windowed_data.shape[1] or current_size <= 0:
                    print(f"Cannot adjust interface size beyond data dimensions. Stopping.")
                    break

        encoder_model_filename = f"{config['save_encoder']}_{column}.keras"
        decoder_model_filename = f"{config['save_decoder']}_{column}.keras"
        autoencoder_manager.save_encoder(encoder_model_filename)
        autoencoder_manager.save_decoder(decoder_model_filename)
        print(f"Saved encoder model to {encoder_model_filename}")
        print(f"Saved decoder model to {decoder_model_filename}")

        # Perform unwindowing of the decoded data once
        reconstructed_data = unwindow_data(pd.DataFrame(decoded_data))

        output_filename = os.path.splitext(config['output_file'])[0] + f"_{column}.csv"
        write_csv(output_filename, reconstructed_data, include_date=config['force_date'], headers=config['headers'])
        print(f"Output written to {output_filename}")

        print(f"Encoder Dimensions: {autoencoder_manager.encoder_model.input_shape} -> {autoencoder_manager.encoder_model.output_shape}")
        print(f"Decoder Dimensions: {autoencoder_manager.decoder_model.input_shape} -> {autoencoder_manager.decoder_model.output_shape}")

    # Save final configuration and debug information
    end_time = time.time()
    execution_time = end_time - start_time
    debug_info = {
        'execution_time': execution_time,
        'encoder': encoder_plugin.get_debug_info(),
        'decoder': decoder_plugin.get_debug_info(),
        'mse': mse,
        'mae': mae
    }

    # save debug info
    if 'save_log' in config:
        if config['save_log'] != None:
            save_debug_info(debug_info, config['save_log'])
            print(f"Debug info saved to {config['save_log']}.")

    # remote log debug info and config
    if 'remote_log' in config:
        if config['remote_log'] != None:
            remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
            print(f"Debug info saved to {config['remote_log']}.")

    print(f"Execution time: {execution_time} seconds")

def load_and_evaluate_encoder(config, encoder_plugin):
    # Load the encoder model
    encoder_plugin.load(config['load_encoder'])

    # Load the input data
    processed_data = process_data(config)
    column = list(processed_data.keys())[0]
    windowed_data = processed_data[column]
    
    # Encode the data
    encoded_data = encoder_plugin.encode(windowed_data)

    # Save the encoded data to CSV
    evaluate_filename = config['evaluate_encoder']
    np.savetxt(evaluate_filename, encoded_data, delimiter=",")
    print(f"Encoded data saved to {evaluate_filename}")

def load_and_evaluate_decoder(config, decoder_plugin):
    # Load the decoder model
    decoder_plugin.load(config['load_decoder'])

    # Load the input data
    processed_data = process_data(config)
    column = list(processed_data.keys())[0]
    windowed_data = processed_data[column]

    # Decode the data
    decoded_data = decoder_plugin.decode(windowed_data)

    # Save the decoded data to CSV
    evaluate_filename = config['evaluate_decoder']
    np.savetxt(evaluate_filename, decoded_data, delimiter=",")
    print(f"Decoded data saved to {evaluate_filename}")
