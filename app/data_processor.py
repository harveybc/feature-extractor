import sys
import requests
import numpy as np
from app.data_handler import load_csv, write_csv, sliding_window
from app.plugin_loader import load_encoder_decoder_plugins

def train_autoencoder(encoder, decoder, data, mse_threshold, initial_size, step_size, incremental_search, epochs):
    """
    Train an autoencoder with decreasing interface size until the mse_threshold is reached.

    Args:
        encoder (object): The encoder plugin instance.
        decoder (object): The decoder plugin instance.
        data (np.array): The data to train on.
        mse_threshold (float): Maximum mean squared error threshold.
        initial_size (int): Initial size of the encoder/decoder interface.
        step_size (int): Step size to adjust the interface size.
        incremental_search (bool): If true, incrementally increase the interface size.

    Returns:
        object: Trained encoder and decoder instances.
    """
    current_size = initial_size
    current_mse = float('inf')
    print(f"Training autoencoder with initial size {current_size}...")
    
    while current_size > 0 and ((current_mse > mse_threshold) if not incremental_search else (current_mse < mse_threshold)):
        encoder.configure_size(input_dim=data.shape[1], encoding_dim=current_size)
        decoder.configure_size(encoding_dim=current_size, output_dim=data.shape[1])
        encoder.train(data)
        encoded_data = encoder.encode(data)
        decoder.train(encoded_data, data)
        
        encoded_data = encoder.encode(data)
        decoded_data = decoder.decode(encoded_data)
        current_mse = encoder.calculate_mse(data, decoded_data)
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
    
    return encoder, decoder

def process_data(config):
    """
    Process the data using the specified encoder and decoder plugins.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: Processed data and debug information.
    """
    data = load_csv(config['csv_file'], headers=config['headers'])
    print(f"Data loaded: {data.shape[0]} rows and {data.shape[1]} columns.")
    
    if config['force_date']:
        data.index = pd.to_datetime(data.index)

    print(f"Data types:\n {data.dtypes}")

    debug_info = {}

    encoder_name = config.get('encoder_plugin', 'default_encoder')
    decoder_name = config.get('decoder_plugin', 'default_decoder')
    Encoder, encoder_params, Decoder, decoder_params = load_encoder_decoder_plugins(encoder_name, decoder_name)

    for column in data.columns:
        print(f"Processing column: {column}")
        column_data = data[[column]].values.astype(np.float64)
        windowed_data = sliding_window(column_data, config['window_size'])
        print(f"Windowed data shape: {windowed_data.shape}")

        trained_encoder, trained_decoder = train_autoencoder(
            Encoder(), Decoder(), windowed_data, config['mse_threshold'], 
            config['initial_encoding_dim'], config['encoding_step_size'], 
            config['incremental_search'], config['epochs']
        )

        encoder_model_filename = f"{config['save_encoder_path']}_{column}.keras"
        decoder_model_filename = f"{config['save_decoder_path']}_{column}.keras"
        trained_encoder.save(encoder_model_filename)
        trained_decoder.save(decoder_model_filename)
        print(f"Saved encoder model to {encoder_model_filename}")
        print(f"Saved decoder model to {decoder_model_filename}")

        encoded_data = trained_encoder.encode(windowed_data)
        decoded_data = trained_decoder.decode(encoded_data)

        mse = trained_encoder.calculate_mse(windowed_data, decoded_data)
        print(f"Mean Squared Error for column {column}: {mse}")
        debug_info[f'mean_squared_error_{column}'] = mse

        # Reshape the decoded data back to the original shape
        reshaped_decoded_data = decoded_data.reshape(-1, config['window_size'])

        # Scale the data back to the range -1 to 1 if necessary
        reshaped_decoded_data = 2 * (reshaped_decoded_data - 0.5)  # Assuming the data was scaled from -1 to 1 to 0 to 1
        
        output_filename = f"{config['csv_output_path']}_{column}.csv"
        write_csv(output_filename, reshaped_decoded_data, include_date=config['force_date'], headers=config['headers'])
        print(f"Output written to {output_filename}")

        # Print the encoder and decoder dimensions
        print(f"Encoder Dimensions: {trained_encoder.model.input_shape} -> {trained_encoder.model.output_shape}")
        print(f"Decoder Dimensions: {trained_decoder.model.input_shape} -> {trained_decoder.model.output_shape}")

    return reshaped_decoded_data, debug_info
