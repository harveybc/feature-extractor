import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from app.autoencoder_manager import AutoencoderManager
from app.data_handler import load_csv, write_csv
from app.reconstruction import unwindow_data
from app.config_handler import save_debug_info, remote_log
from keras.models import Sequential, Model, load_model

def create_sliding_windows(data, window_size):
    """
    Create sliding windows for the entire dataset, returning a 3D array suitable for Conv1D.
    
    Args:
        data (pd.DataFrame): Dataframe containing the time series features.
        window_size (int): The length of the sliding window.
    
    Returns:
        np.ndarray: Array with shape (num_samples, window_size, num_features).
    """
    data_array = data.to_numpy()  # Convert dataframe to numpy array
    num_features = data_array.shape[1]  # Number of columns/features
    num_samples = data_array.shape[0] - window_size + 1  # Calculate the number of sliding windows
    
    # Create a 3D array to store the windows
    windows = np.zeros((num_samples, window_size, num_features))
    
    # Slide the window over the data and create the 3D array
    for i in range(num_samples):
        windows[i] = data_array[i:i+window_size]  # Slice window_size rows across all columns
    
    return windows


def process_data(config):
    print(f"Loading data from CSV file: {config['input_file']}")
    data = load_csv(config['input_file'], headers=config['headers'])
    print(f"Data loaded with shape: {data.shape}")

    window_size = config['window_size']
    print(f"Applying sliding window of size: {window_size}")
    
    # Apply sliding windows to the entire dataset (multi-column)
    windowed_data = create_sliding_windows(data, window_size)
    print(f"Windowed data shape: {windowed_data.shape}")  # Should be (num_samples, window_size, num_features)

    # Now do the same for the validation dataset
    print(f"Loading validation data from CSV file: {config['validation_file']}")
    validation_data = load_csv(config['validation_file'], headers=config['headers'])
    print(f"Validation data loaded with shape: {validation_data.shape}")
    
    # Apply sliding windows to the validation dataset
    windowed_validation_data = create_sliding_windows(validation_data, window_size)
    print(f"Windowed validation data shape: {windowed_validation_data.shape}")

    # Return the processed datasets (windowed)
    return windowed_data, windowed_validation_data

def run_autoencoder_pipeline(config, encoder_plugin, decoder_plugin):
    start_time = time.time()
    
    print("Running process_data...")
    processed_data, validation_data = process_data(config)
    print("Processed data received.")

    mse = 0
    mae = 0
    initial_size = config['initial_size']
    step_size = config['step_size']
    threshold_error = config['threshold_error']
    training_batch_size = config['batch_size']
    epochs = config['epochs']
    incremental_search = config['incremental_search']
    
    current_size = initial_size
    while True:
        print(f"Training with interface size: {current_size}")
        
        autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
        num_channels = processed_data.shape[-1]

        # Build new autoencoder model with the current size
        autoencoder_manager.build_autoencoder(config['window_size'], current_size, config, num_channels)

        # Train the autoencoder model
        autoencoder_manager.train_autoencoder(processed_data, epochs=epochs, batch_size=training_batch_size)

        # Encode and decode the validation data
        encoded_data = autoencoder_manager.encode_data(validation_data)  
        decoded_data = autoencoder_manager.decode_data(encoded_data)

        # Since both the original validation data and decoded data now have the same shape,
        # directly calculate the MSE and MAE without unwindowing.
        # Trim the data to ensure sizes match
        min_size = min(validation_data.shape[0], decoded_data.shape[0])
        validation_trimmed = validation_data[:min_size]
        decoded_trimmed = decoded_data[:min_size]

        # Ensure both trimmed arrays are NumPy arrays
        validation_trimmed = np.asarray(validation_trimmed)
        decoded_trimmed = np.asarray(decoded_trimmed)

        # Calculate the MSE and MAE directly
        mse = autoencoder_manager.calculate_mse(validation_trimmed, decoded_trimmed)
        mae = autoencoder_manager.calculate_mae(validation_trimmed, decoded_trimmed)

        print(f"Mean Squared Error with interface size {current_size}: {mse}")
        print(f"Mean Absolute Error with interface size {current_size}: {mae}")

        if (incremental_search and mae <= threshold_error) or (not incremental_search and mae >= threshold_error):
            print(f"Optimal interface size found: {current_size} with MSE: {mse} and MAE: {mae}")
            break
        else:
            if incremental_search:
                current_size += step_size
            else:
                current_size -= step_size
            if current_size > processed_data.shape[1] or current_size <= 0:
                print(f"Cannot adjust interface size beyond data dimensions. Stopping.")
                break

    encoder_model_filename = f"{config['save_encoder']}.keras"
    decoder_model_filename = f"{config['save_decoder']}.keras"
    autoencoder_manager.save_encoder(encoder_model_filename)
    autoencoder_manager.save_decoder(decoder_model_filename)
    print(f"Saved encoder model to {encoder_model_filename}")
    print(f"Saved decoder model to {decoder_model_filename}")

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

    # Save debug info
    if 'save_log' in config and config['save_log']:
        save_debug_info(debug_info, config['save_log'])
        print(f"Debug info saved to {config['save_log']}.")

    # Remote log debug info and config
    if 'remote_log' in config and config['remote_log']:
        remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
        print(f"Debug info saved to {config['remote_log']}.")

    print(f"Execution time: {execution_time} seconds")



def load_and_evaluate_encoder(config):
    model = load_model(config['load_encoder'])
    print(f"Encoder model loaded from {config['load_encoder']}")
    
    # Load the input data
    processed_data, validation_data = process_data(config)
    
    # Use processed data directly
    windowed_data = processed_data  # This already holds the entire windowed data
    
    # Encode the data
    print(f"Encoding data with shape: {windowed_data.shape}")
    encoded_data = model.predict(windowed_data)
    print(f"Encoded data shape: {encoded_data.shape}")
    
    # Adjust the reshaping logic
    # Instead of reshaping based on model.output_shape[2], reshape it to maintain the original encoded shape
    # Ensure we flatten it only if necessary or handle it based on the next step in the process
    # Example: Flatten the second and third dimensions if needed
    if len(encoded_data.shape) == 3:
        # Flatten the last two dimensions
        encoded_data = encoded_data.reshape(encoded_data.shape[0], -1)  # (27798, 128)
    
    # Save the encoded data to CSV
    evaluate_filename = config['evaluate_encoder']
    np.savetxt(evaluate_filename, encoded_data, delimiter=",")
    print(f"Encoded data saved to {evaluate_filename}")




def load_and_evaluate_decoder(config):
    model = load_model(config['load_decoder'])
    print(f"Decoder model loaded from {config['load_decoder']}")
    # Load the input data
    processed_data = process_data(config)
    column = list(processed_data.keys())[0]
    windowed_data = processed_data[column]
    # Decode the data
    print(f"Decoding data with shape: {windowed_data.shape}")
    decoded_data = model.predict(windowed_data)
    print(f"Decoded data shape: {decoded_data.shape}")
    # Check if the decoded data needs reshaping
    if len(decoded_data.shape) == 3:
        decoded_data = decoded_data.reshape(decoded_data.shape[0], decoded_data.shape[2])
    # Save the encoded data to CSV
    evaluate_filename = config['evaluate_decoder']
    np.savetxt(evaluate_filename, decoded_data, delimiter=",")
    print(f"Decoded data saved to {evaluate_filename}")
