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
    import numpy as np
    start_time = time.time()
    
    print("Running process_data...")
    processed_data, validation_data = process_data(config)
    print("Processed data received.")

    # Truncate validation data to have at most as many rows as training data
    if validation_data.shape[0] > processed_data.shape[0]:
        print(f"[run_autoencoder_pipeline] Truncating validation data from {validation_data.shape[0]} rows to match training data rows: {processed_data.shape[0]}")
        validation_data = validation_data[:processed_data.shape[0]]
    
    # Get the encoder plugin name (lowercase)
    encoder_plugin_name = config.get('encoder_plugin', '').lower()
    
    # For sequential plugins (LSTM/Transformer) without sliding windows, expand dims at axis 1.
    if not config.get('use_sliding_windows', True):
        if encoder_plugin_name in ['lstm', 'transformer']:
            print("[run_autoencoder_pipeline] Detected sequential plugin (LSTM/Transformer) without sliding windows; expanding dimension at axis 1.")
            processed_data = np.expand_dims(processed_data, axis=1)  # becomes (samples, 1, features)
            validation_data = np.expand_dims(validation_data, axis=1)
            config['original_feature_size'] = processed_data.shape[2]
        elif encoder_plugin_name == 'cnn':
            print("[run_autoencoder_pipeline] Detected CNN plugin without sliding windows; expanding dimension at axis 1.")
            processed_data = np.expand_dims(processed_data, axis=1)
            validation_data = np.expand_dims(validation_data, axis=1)
            config['original_feature_size'] = validation_data.shape[2]
        else:
            config['original_feature_size'] = validation_data.shape[1]
            print(f"[run_autoencoder_pipeline] Set original_feature_size: {config['original_feature_size']}")
    
    # Determine input_size:
    # - If sliding windows are used:
    #     For CNN plugins, input_size is a tuple (window_size, num_features);
    #     Otherwise, input_size equals window_size.
    # - For sequential plugins (LSTM/Transformer) without sliding windows, input_size is the number of features.
    if config.get('use_sliding_windows', False):
        if encoder_plugin_name == 'cnn':
            input_size = (config['window_size'], processed_data.shape[-1])
        else:
            input_size = config['window_size']
    else:
        if encoder_plugin_name in ['lstm', 'transformer']:
            input_size = config['original_feature_size']
        else:
            input_size = processed_data.shape[1]
    
    initial_size = config['initial_size']
    step_size = config['step_size']
    threshold_error = config['threshold_error']
    training_batch_size = config['batch_size']
    epochs = config['epochs']
    incremental_search = config['incremental_search']
    
    current_size = initial_size
    
    while True:
        print(f"Training with interface size: {current_size}")
        
        # num_channels is taken from the last dimension of processed_data.
        num_channels = processed_data.shape[-1]
        
        from app.autoencoder_manager import AutoencoderManager
        autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
        autoencoder_manager.build_autoencoder(input_size, current_size, config, num_channels)
        autoencoder_manager.train_autoencoder(processed_data, epochs=epochs, batch_size=training_batch_size, config=config)
        
        training_mse, training_mae = autoencoder_manager.evaluate(processed_data, "Training", config)
        print(f"Training Mean Squared Error with interface size {current_size}: {training_mse}")
        print(f"Training Mean Absolute Error with interface size {current_size}: {training_mae}")
        
        validation_mse, validation_mae = autoencoder_manager.evaluate(validation_data, "Validation", config)
        print(f"Validation Mean Squared Error with interface size {current_size}: {validation_mse}")
        print(f"Validation Mean Absolute Error with interface size {current_size}: {validation_mae}")
        
        if (incremental_search and validation_mae <= threshold_error) or (not incremental_search and validation_mae >= threshold_error):
            print(f"Optimal interface size found: {current_size} with Validation MSE: {validation_mse} and Validation MAE: {validation_mae}")
            break
        else:
            if incremental_search:
                current_size += step_size
            else:
                current_size -= step_size
            if current_size > processed_data.shape[1] or current_size <= 0:
                print("Cannot adjust interface size beyond data dimensions. Stopping.")
                break

    encoder_model_filename = f"{config['save_encoder']}.keras"
    decoder_model_filename = f"{config['save_decoder']}.keras"
    autoencoder_manager.save_encoder(encoder_model_filename)
    autoencoder_manager.save_decoder(decoder_model_filename)
    print(f"Saved encoder model to {encoder_model_filename}")
    print(f"Saved decoder model to {decoder_model_filename}")

    end_time = time.time()
    execution_time = end_time - start_time
    debug_info = {
        'execution_time': execution_time,
        'encoder': encoder_plugin.get_debug_info(),
        'decoder': decoder_plugin.get_debug_info(),
        'mse': validation_mse,
        'mae': validation_mae
    }

    from app.config_handler import save_debug_info, remote_log
    if 'save_log' in config and config['save_log']:
        save_debug_info(debug_info, config['save_log'])
        print(f"Debug info saved to {config['save_log']}.")
    
    if 'remote_log' in config and config['remote_log']:
        remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
        print(f"Debug info saved to {config['remote_log']}.")
    
    print(f"Execution time: {execution_time} seconds")




def load_and_evaluate_encoder(config):
    """
    Load and evaluate a pre-trained encoder with input data.
    """
    if config.get('encoder_plugin', '').lower() == 'transformer':
        import tensorflow as tf
        from keras_multi_head import MultiHeadAttention as OriginalMultiHeadAttention
        from tensorflow.keras.layers import LayerNormalization
        from tensorflow.keras.activations import gelu
        # Import positional_encoding from your plugin without modifying it.
        from app.plugins.encoder_plugin_transformer import positional_encoding

        # Patched MultiHeadAttention to handle 'head_num' correctly.
        class PatchedMultiHeadAttention(OriginalMultiHeadAttention):
            @classmethod
            def from_config(cls, config):
                head_num = config.pop("head_num", None)
                if head_num is None:
                    head_num = 8  # Default value; adjust if needed.
                return cls(head_num, **config)

        custom_objects = {
            'MultiHeadAttention': PatchedMultiHeadAttention,
            'LayerNormalization': LayerNormalization,
            'gelu': gelu,
            # Supply the missing function so that the Lambda layer finds it.
            'positional_encoding': positional_encoding
        }
        model = load_model(config['load_encoder'], custom_objects=custom_objects)
    else:
        model = load_model(config['load_encoder'])
    print(f"Encoder model loaded from {config['load_encoder']}")

    # Load the input data.
    data = load_csv(
        file_path=config['input_file'],
        headers=config.get('headers', False),
        force_date=config.get('force_date', False)
    )
    
    # Extract the original date information.
    original_dates = None
    if config.get('force_date', False) and data.index.name is not None:
        # Reset the index so that the date becomes a column.
        original_dates = data.index.to_series().rename("DATE_TIME").reset_index(drop=True)
    elif "DATE_TIME" in data.columns:
        original_dates = data["DATE_TIME"].reset_index(drop=True)
    
    # Process data based on whether sliding windows are used.
    if config.get('use_sliding_windows', True):
        window_size = config['window_size']
        print(f"Creating sliding windows of size: {window_size}")
        processed_data = create_sliding_windows(data, window_size)
        print(f"Processed data shape for sliding windows: {processed_data.shape}")
    else:
        encoder_plugin_name = config.get('encoder_plugin', '').lower()
        if encoder_plugin_name in ['lstm', 'transformer']:
            print("[load_and_evaluate_encoder] Detected sequential plugin (LSTM/Transformer) without sliding windows; expanding dimension at axis 1.")
            processed_data = np.expand_dims(data.to_numpy(), axis=1)
        else:
            print("Reshaping data to match encoder input shape.")
            processed_data = np.expand_dims(data.to_numpy(), axis=-1)
        print(f"Processed data shape without sliding windows: {processed_data.shape}")

    # Predict using the encoder.
    print(f"Encoding data with shape: {processed_data.shape}")
    encoded_data = model.predict(processed_data, verbose=1)
    print(f"Encoded data shape: {encoded_data.shape}")

    # Flatten 3D encoded data to 2D for saving.
    if len(encoded_data.shape) == 3:
        num_samples, dim1, dim2 = encoded_data.shape
        encoded_data_reshaped = encoded_data.reshape(num_samples, dim1 * dim2)
        print(f"Reshaped encoded data to 2D: {encoded_data_reshaped.shape}")
    elif len(encoded_data.shape) == 2:
        encoded_data_reshaped = encoded_data
    else:
        raise ValueError(f"Unexpected encoded_data shape: {encoded_data.shape}")

    if config.get('evaluate_encoder'):
        print(f"Saving encoded data to {config['evaluate_encoder']}")
        encoded_df = pd.DataFrame(encoded_data_reshaped)
        # Prepend the DATE_TIME column if available and if row counts match.
        if original_dates is not None and len(original_dates) == encoded_df.shape[0]:
            encoded_df.insert(0, "DATE_TIME", original_dates)
        else:
            print("Warning: Original date information not available or row count mismatch.")
        
        # Rename the feature columns.
        if "DATE_TIME" in encoded_df.columns:
            new_columns = ["DATE_TIME"] + [f"feature_{i}" for i in range(encoded_df.shape[1] - 1)]
        else:
            new_columns = [f"feature_{i}" for i in range(encoded_df.shape[1])]
        encoded_df.columns = new_columns
        
        encoded_df.to_csv(config['evaluate_encoder'], index=False)
        print(f"Encoded data saved to {config['evaluate_encoder']}")



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



