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


# app/data_processor.py

def process_data(config):
    """
    Process the data based on the configuration.
    
    Args:
        config (dict): Configuration dictionary with parameters for processing.
    
    Returns:
        tuple: Processed training and validation datasets.
    """
    print(f"Loading data from CSV file: {config['input_file']}")
    data = load_csv(
        file_path=config['input_file'],
        headers=config.get('headers', False),
        force_date=config.get('force_date', False)
    )
    print(f"Data loaded with shape: {data.shape}")

    if config['use_sliding_windows']:
        window_size = config['window_size']
        print(f"Applying sliding window of size: {window_size}")

        # Apply sliding windows to the entire dataset (multi-column)
        processed_data = create_sliding_windows(data, window_size)
        print(f"Windowed data shape: {processed_data.shape}")  # Should be (num_samples, window_size, num_features)
    else:
        print("Skipping sliding windows. Data will be fed row-by-row.")
        # Use data row-by-row as a NumPy array
        processed_data = data.to_numpy()
        print(f"Processed data shape: {processed_data.shape}")  # Should be (num_samples, num_features)

    print(f"Loading validation data from CSV file: {config['validation_file']}")
    validation_data = load_csv(
        file_path=config['validation_file'],
        headers=config.get('headers', False),
        force_date=config.get('force_date', False)
    )
    print(f"Validation data loaded with shape: {validation_data.shape}")

    if config['use_sliding_windows']:
        # Apply sliding windows to the validation dataset
        windowed_validation_data = create_sliding_windows(validation_data, config['window_size'])
        print(f"Windowed validation data shape: {windowed_validation_data.shape}")
    else:
        print("Skipping sliding windows for validation data. Data will be fed row-by-row.")
        # Use validation data row-by-row as a NumPy array
        windowed_validation_data = validation_data.to_numpy()
        print(f"Validation processed shape: {windowed_validation_data.shape}")

    return processed_data, windowed_validation_data



def run_autoencoder_pipeline(config, encoder_plugin, decoder_plugin):
    import time
    start_time = time.time()
    
    print("Running process_data...")
    processed_data, validation_data = process_data(config)
    print("Processed data received.")

    # Dynamically set original feature size after data is processed
    if not config.get('use_sliding_windows', True):
        config['original_feature_size'] = validation_data.shape[1]
        print(f"[run_autoencoder_pipeline] Set original_feature_size: {config['original_feature_size']}")

    mse_train = 0
    mae_train = 0
    mse_val = 0
    mae_val = 0
    initial_size = config['initial_size']
    step_size = config['step_size']
    threshold_error = config['threshold_error']
    training_batch_size = config['batch_size']
    epochs = config['epochs']
    incremental_search = config['incremental_search']
    
    current_size = initial_size
    input_size = config['window_size'] if config['use_sliding_windows'] else processed_data.shape[1]

    while True:
        print(f"Training with interface size: {current_size}")
        
        autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
        num_channels = processed_data.shape[-1]

        # Build new autoencoder model with the current input size
        autoencoder_manager.build_autoencoder(input_size, current_size, config, num_channels)

        # Train the autoencoder model
        autoencoder_manager.train_autoencoder(processed_data, epochs=epochs, batch_size=training_batch_size, config=config)

        # Evaluate the autoencoder on training data
        encoded_train = autoencoder_manager.encode_data(processed_data, config)
        decoded_train = autoencoder_manager.decode_data(encoded_train, config)

        # Evaluate the autoencoder on validation data
        encoded_val = autoencoder_manager.encode_data(validation_data, config)
        decoded_val = autoencoder_manager.decode_data(encoded_val, config)

        # Ensure trimmed arrays are NumPy arrays
        train_trimmed = np.asarray(processed_data[:decoded_train.shape[0]])
        decoded_train_trimmed = np.asarray(decoded_train)
        val_trimmed = np.asarray(validation_data[:decoded_val.shape[0]])
        decoded_val_trimmed = np.asarray(decoded_val)

        # Handle potential shape mismatch for non-sliding window cases
        if not config.get('use_sliding_windows', True):
            decoded_train_trimmed = decoded_train_trimmed.reshape(train_trimmed.shape)
            decoded_val_trimmed = decoded_val_trimmed.reshape(val_trimmed.shape)

        # Calculate the training MSE and MAE
        mse_train = autoencoder_manager.calculate_mse(train_trimmed, decoded_train_trimmed, config)
        mae_train = autoencoder_manager.calculate_mae(train_trimmed, decoded_train_trimmed, config)

        # Calculate the validation MSE and MAE
        mse_val = autoencoder_manager.calculate_mse(val_trimmed, decoded_val_trimmed, config)
        mae_val = autoencoder_manager.calculate_mae(val_trimmed, decoded_val_trimmed, config)

        # Print results
        print(f"Training Mean Squared Error with interface size {current_size}: {mse_train}")
        print(f"Training Mean Absolute Error with interface size {current_size}: {mae_train}")
        print(f"Validation Mean Squared Error with interface size {current_size}: {mse_val}")
        print(f"Validation Mean Absolute Error with interface size {current_size}: {mae_val}")

        if (incremental_search and mae_val <= threshold_error) or (not incremental_search and mae_val >= threshold_error):
            print(f"Optimal interface size found: {current_size} with Validation MSE: {mse_val} and Validation MAE: {mae_val}")
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
        'mse_train': mse_train,
        'mae_train': mae_train,
        'mse_val': mse_val,
        'mae_val': mae_val
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



# app/data_processor.py

def load_and_evaluate_encoder(config):
    model = load_model(config['load_encoder'])
    print(f"Encoder model loaded from {config['load_encoder']}")

    # Load the input data with headers and date based on config
    data = load_csv(
        file_path=config['input_file'],
        headers=config.get('headers', False),
        force_date=config.get('force_date', False)
    )

    # Apply sliding window
    window_size = config['window_size']
    windowed_data = create_sliding_windows(data, window_size)

    print(f"Encoding data with shape: {windowed_data.shape}")
    encoded_data = model.predict(windowed_data)
    print(f"Encoded data shape: {encoded_data.shape}")

    # Reshape encoded_data from (samples, 32, 8) to (samples, 256)
    if len(encoded_data.shape) == 3:
        samples, dim1, dim2 = encoded_data.shape
        encoded_data = encoded_data.reshape(samples, dim1 * dim2)
        print(f"Reshaped encoded data to: {encoded_data.shape}")
    elif len(encoded_data.shape) != 2:
        raise ValueError(f"Unexpected encoded_data shape: {encoded_data.shape}")

    if config.get('force_date', False):
        # Extract corresponding dates for each window
        dates = data.index[window_size - 1:]
        # Create a DataFrame with dates and encoded features
        encoded_df = pd.DataFrame(encoded_data, index=dates)
        encoded_df.index.name = 'date'
    else:
        # Create a DataFrame without dates
        encoded_df = pd.DataFrame(encoded_data)

    # Assign headers for encoded features, e.g., 'encoded_feature_1', 'encoded_feature_2', etc.
    feature_names = [f'encoded_feature_{i+1}' for i in range(encoded_data.shape[1])]
    encoded_df.columns = feature_names

    # Save the encoded data to CSV using the write_csv function
    evaluate_filename = config['evaluate_encoder']
    write_csv(
        file_path=evaluate_filename,
        data=encoded_df,
        include_date=config.get('force_date', False),
        headers=True,  # Always include headers for encoded features
        force_date=config.get('force_date', False)
    )
    print(f"Encoded data saved to {evaluate_filename}")




# app/data_processor.py

def load_and_evaluate_decoder(config):
    model = load_model(config['load_decoder'])
    print(f"Decoder model loaded from {config['load_decoder']}")

    # Load the input data with headers and date based on config
    data = load_csv(
        file_path=config['input_file'],
        headers=config.get('headers', False),
        force_date=config.get('force_date', False)
    )

    # Apply sliding window
    window_size = config['window_size']
    windowed_data = create_sliding_windows(data, window_size)

    print(f"Decoding data with shape: {windowed_data.shape}")
    decoded_data = model.predict(windowed_data)
    print(f"Decoded data shape: {decoded_data.shape}")

    # Reshape decoded_data from (samples, 32, 8) to (samples, 256)
    if len(decoded_data.shape) == 3:
        samples, dim1, dim2 = decoded_data.shape
        decoded_data = decoded_data.reshape(samples, dim1 * dim2)
        print(f"Reshaped decoded data to: {decoded_data.shape}")
    elif len(decoded_data.shape) != 2:
        raise ValueError(f"Unexpected decoded_data shape: {decoded_data.shape}")

    if config.get('force_date', False):
        # Extract corresponding dates for each window
        dates = data.index[window_size - 1:]
        # Create a DataFrame with dates and decoded features
        decoded_df = pd.DataFrame(decoded_data, index=dates)
        decoded_df.index.name = 'date'
    else:
        # Create a DataFrame without dates
        decoded_df = pd.DataFrame(decoded_data)

    # Assign headers for decoded features, e.g., 'decoded_feature_1', 'decoded_feature_2', etc.
    feature_names = [f'decoded_feature_{i+1}' for i in range(decoded_data.shape[1])]
    decoded_df.columns = feature_names

    # Save the decoded data to CSV using the write_csv function
    evaluate_filename = config['evaluate_decoder']
    write_csv(
        file_path=evaluate_filename,
        data=decoded_df,
        include_date=config.get('force_date', False),
        headers=True,  # Always include headers for decoded features
        force_date=config.get('force_date', False)
    )
    print(f"Decoded data saved to {evaluate_filename}")



