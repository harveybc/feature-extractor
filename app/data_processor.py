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
    """
    Process the data based on the configuration.
    
    If future_shift > 0, returns:
        ((training_input, training_target), (validation_input, validation_target))
    Else, returns:
        (processed_data, windowed_validation_data)
    
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
        processed_data = create_sliding_windows(data, window_size)
        print(f"Windowed data shape: {processed_data.shape}")  # (num_samples, window_size, num_features)
    else:
        print("Skipping sliding windows. Data will be fed row-by-row.")
        processed_data = data.to_numpy()
        print(f"Processed data shape: {processed_data.shape}")  # (num_samples, num_features)

    print(f"Loading validation data from CSV file: {config['validation_file']}")
    validation_data = load_csv(
        file_path=config['validation_file'],
        headers=config.get('headers', False),
        force_date=config.get('force_date', False)
    )
    print(f"Validation data loaded with shape: {validation_data.shape}")

    if config['use_sliding_windows']:
        windowed_validation_data = create_sliding_windows(validation_data, config['window_size'])
        print(f"Windowed validation data shape: {windowed_validation_data.shape}")
    else:
        print("Skipping sliding windows for validation data. Data will be fed row-by-row.")
        windowed_validation_data = validation_data.to_numpy()
        print(f"Validation processed shape: {windowed_validation_data.shape}")

    # --- FUTURE_SHIFT LOGIC ---
    future_shift = config.get('future_shift', 0)
    if future_shift > 0:
        print(f"Applying future_shift: {future_shift}")
        # Process training data with future shift
        if config['use_sliding_windows']:
            windows = processed_data  # shape: (num_windows, window_size, num_features)
            valid_samples = windows.shape[0] - future_shift
            if valid_samples <= 0:
                raise ValueError("The combination of window_size and future_shift exceeds the available data rows.")
            training_input = windows[:valid_samples]
            # For window i, target is taken from original data at index (window_size - 1 + future_shift + i)
            training_target = data.to_numpy()[config['window_size'] - 1 + future_shift : config['window_size'] - 1 + future_shift + valid_samples]
            print(f"Future_shift trimming (training):")
            print(f"  Original training windows: {windows.shape[0]} -> Valid samples: {valid_samples}")
            print(f"  Training input shape: {training_input.shape}, Training target shape: {training_target.shape}")
        else:
            raw = processed_data  # shape: (N, num_features)
            valid_samples = raw.shape[0] - future_shift
            if valid_samples <= 0:
                raise ValueError("future_shift exceeds the number of data rows.")
            training_input = raw[:valid_samples]
            training_target = raw[future_shift:]
            print(f"Future_shift trimming (training):")
            print(f"  Original rows: {raw.shape[0]} -> Valid samples: {valid_samples}")
            print(f"  Training input shape: {training_input.shape}, Training target shape: {training_target.shape}")
            
        # Process validation data similarly
        if config['use_sliding_windows']:
            val_windows = windowed_validation_data  # shape: (num_windows_val, window_size, num_features)
            valid_samples_val = val_windows.shape[0] - future_shift
            if valid_samples_val <= 0:
                raise ValueError("The combination of window_size and future_shift exceeds the available validation data rows.")
            validation_input = val_windows[:valid_samples_val]
            validation_target = validation_data.to_numpy()[config['window_size'] - 1 + future_shift : config['window_size'] - 1 + future_shift + valid_samples_val]
            print(f"Future_shift trimming (validation):")
            print(f"  Original validation windows: {val_windows.shape[0]} -> Valid samples: {valid_samples_val}")
            print(f"  Validation input shape: {validation_input.shape}, Validation target shape: {validation_target.shape}")
        else:
            raw_val = windowed_validation_data  # shape: (N_val, num_features)
            valid_samples_val = raw_val.shape[0] - future_shift
            if valid_samples_val <= 0:
                raise ValueError("future_shift exceeds the number of validation data rows.")
            validation_input = raw_val[:valid_samples_val]
            validation_target = raw_val[future_shift:]
            print(f"Future_shift trimming (validation):")
            print(f"  Original validation rows: {raw_val.shape[0]} -> Valid samples: {valid_samples_val}")
            print(f"  Validation input shape: {validation_input.shape}, Validation target shape: {validation_target.shape}")

        return (training_input, training_target), (validation_input, validation_target)
    else:
        return processed_data, windowed_validation_data


def run_autoencoder_pipeline(config, encoder_plugin, decoder_plugin):
    import time
    import numpy as np
    start_time = time.time()
    
    print("Running process_data...")
    future_shift = config.get('future_shift', 0)
    if future_shift > 0:
        (training_input, training_target), (validation_input, validation_target) = process_data(config)
        print("Processed data received with future_shift > 0.")
    else:
        training_input, validation_input = process_data(config)
        training_target = training_input  # pure autoencoder reconstruction
        validation_target = validation_input
        print("Processed data received with future_shift = 0 (pure autoencoder).")

    # Truncate validation data if necessary
    if training_input.shape[0] > validation_input.shape[0]:
        print(f"[run_autoencoder_pipeline] Truncating training data from {training_input.shape[0]} rows to match validation data rows: {validation_input.shape[0]}")
        training_input = training_input[:validation_input.shape[0]]
        training_target = training_target[:validation_input.shape[0]]
    elif validation_input.shape[0] > training_input.shape[0]:
        print(f"[run_autoencoder_pipeline] Truncating validation data from {validation_input.shape[0]} rows to match training data rows: {training_input.shape[0]}")
        validation_input = validation_input[:training_input.shape[0]]
        validation_target = validation_target[:training_input.shape[0]]
    
    # Get the encoder plugin name (lowercase)
    encoder_plugin_name = config.get('encoder_plugin', '').lower()
    
    # For sequential plugins (LSTM/Transformer) without sliding windows, expand dims at axis 1.
    if not config.get('use_sliding_windows', True):
        if encoder_plugin_name in ['lstm', 'transformer']:
            print("[run_autoencoder_pipeline] Detected sequential plugin (LSTM/Transformer) without sliding windows; expanding dimension at axis 1.")
            training_input = np.expand_dims(training_input, axis=1)
            validation_input = np.expand_dims(validation_input, axis=1)
            config['original_feature_size'] = training_input.shape[2]
        elif encoder_plugin_name == 'cnn':
            print("[run_autoencoder_pipeline] Detected CNN plugin without sliding windows; expanding dimension at axis 1.")
            training_input = np.expand_dims(training_input, axis=1)
            validation_input = np.expand_dims(validation_input, axis=1)
            config['original_feature_size'] = validation_input.shape[2]
        else:
            config['original_feature_size'] = validation_input.shape[1]
            print(f"[run_autoencoder_pipeline] Set original_feature_size: {config['original_feature_size']}")
    
    # Determine input_size.
    if config.get('use_sliding_windows', False):
        input_size = config['window_size']
    else:
        if encoder_plugin_name in ['lstm', 'transformer']:
            input_size = config['original_feature_size']
        else:
            input_size = training_input.shape[1]
    
    initial_size = config['initial_size']
    step_size = config['step_size']
    threshold_error = config['threshold_error']
    training_batch_size = config['batch_size']
    epochs = config['epochs']
    incremental_search = config['incremental_search']
    
    current_size = initial_size
    
    while True:
        print(f"Training with interface size: {current_size}")
        num_channels = training_input.shape[-1]
        
        from app.autoencoder_manager import AutoencoderManager
        autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
        autoencoder_manager.build_autoencoder(input_size, current_size, config, num_channels)
        autoencoder_manager.train_autoencoder(training_input, target_data=training_target,
                                              epochs=epochs, batch_size=training_batch_size, config=config)
        
        training_mse, training_mae = autoencoder_manager.evaluate(training_input, "Training", config, target_data=training_target)
        print(f"Training Mean Squared Error with interface size {current_size}: {training_mse}")
        print(f"Training Mean Absolute Error with interface size {current_size}: {training_mae}")
        
        validation_mse, validation_mae = autoencoder_manager.evaluate(validation_input, "Validation", config, target_data=validation_target)
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
            if current_size > training_input.shape[1] or current_size <= 0:
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
        from app.plugins.encoder_plugin_transformer import positional_encoding
        class PatchedMultiHeadAttention(OriginalMultiHeadAttention):
            @classmethod
            def from_config(cls, config):
                head_num = config.pop("head_num", None)
                if head_num is None:
                    head_num = 8
                return cls(head_num, **config)
        custom_objects = {
            'MultiHeadAttention': PatchedMultiHeadAttention,
            'LayerNormalization': LayerNormalization,
            'gelu': gelu,
            'positional_encoding': positional_encoding
        }
        model = load_model(config['load_encoder'], custom_objects=custom_objects)
    else:
        model = load_model(config['load_encoder'])
    print(f"Encoder model loaded from {config['load_encoder']}")

    data = load_csv(
        file_path=config['input_file'],
        headers=config.get('headers', False),
        force_date=config.get('force_date', False)
    )
    
    original_dates = None
    if config.get('force_date', False) and data.index.name is not None:
        original_dates = data.index.to_series().rename("DATE_TIME").reset_index(drop=True)
    elif "DATE_TIME" in data.columns:
        original_dates = data["DATE_TIME"].reset_index(drop=True)
    
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

    print(f"Encoding data with shape: {processed_data.shape}")
    encoded_data = model.predict(processed_data, verbose=1)
    print(f"Encoded data shape: {encoded_data.shape}")

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
        if original_dates is not None and len(original_dates) == encoded_df.shape[0]:
            encoded_df.insert(0, "DATE_TIME", original_dates)
        else:
            print("Warning: Original date information not available or row count mismatch.")
        if "DATE_TIME" in encoded_df.columns:
            new_columns = ["DATE_TIME"] + [f"feature_{i}" for i in range(encoded_df.shape[1] - 1)]
        else:
            new_columns = [f"feature_{i}" for i in range(encoded_df.shape[1])]
        encoded_df.columns = new_columns
        encoded_df.to_csv(config['evaluate_encoder'], index=False)
        print(f"Encoded data saved to {config['evaluate_encoder']}")


def load_and_evaluate_decoder(config):
    model = load_model(config['load_decoder'])
    print(f"Decoder model loaded from {config['load_decoder']}")
    data = load_csv(
        file_path=config['input_file'],
        headers=config.get('headers', False),
        force_date=config.get('force_date', False)
    )
    window_size = config['window_size']
    windowed_data = create_sliding_windows(data, window_size)
    print(f"Decoding data with shape: {windowed_data.shape}")
    decoded_data = model.predict(windowed_data)
    print(f"Decoded data shape: {decoded_data.shape}")
    if len(decoded_data.shape) == 3:
        samples, dim1, dim2 = decoded_data.shape
        decoded_data = decoded_data.reshape(samples, dim1 * dim2)
        print(f"Reshaped decoded data to: {decoded_data.shape}")
    elif len(decoded_data.shape) != 2:
        raise ValueError(f"Unexpected decoded_data shape: {decoded_data.shape}")
    if config.get('force_date', False):
        dates = data.index[window_size - 1:]
        decoded_df = pd.DataFrame(decoded_data, index=dates)
        decoded_df.index.name = 'date'
    else:
        decoded_df = pd.DataFrame(decoded_data)
    feature_names = [f'decoded_feature_{i+1}' for i in range(decoded_data.shape[1])]
    decoded_df.columns = feature_names
    evaluate_filename = config['evaluate_decoder']
    write_csv(
        file_path=evaluate_filename,
        data=decoded_df,
        include_date=config.get('force_date', False),
        headers=True,
        force_date=config.get('force_date', False)
    )
    print(f"Decoded data saved to {evaluate_filename}")
