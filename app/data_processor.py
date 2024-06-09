# data_processor.py
import sys
import requests
from app.data_handler import load_csv, write_csv, sliding_window
from app.plugin_loader import load_encoder_decoder_plugins

def train_autoencoder(encoder, decoder, data, max_error, initial_size, step_size, incremental_search):
    """
    Train an autoencoder with adjusting interface size until the max_error is reached.

    Args:
        encoder (object): The encoder plugin instance.
        decoder (object): The decoder plugin instance.
        data (np.array): The data to train on.
        max_error (float): Maximum mean squared error threshold.
        initial_size (int): Initial size of the encoder/decoder interface.
        step_size (int): Step size to adjust the interface size.
        incremental_search (bool): Whether to use incremental search.

    Returns:
        tuple: Trained encoder and decoder instances.
    """
    current_size = initial_size
    current_mse = float('inf')
    while (current_size < data.shape[1] if incremental_search else current_size > 0) and current_mse > max_error:
        print(f"Configuring encoder and decoder with interface size {current_size}...")
        encoder.configure_size(input_dim=data.shape[1], encoding_dim=current_size)
        decoder.configure_size(encoding_dim=current_size, output_dim=data.shape[1])
        
        print(f"Training autoencoder with interface size {current_size}...")
        encoder.train(data)
        encoded_data = encoder.encode(data)
        decoder.train(encoded_data, data)
        
        decoded_data = decoder.decode(encoded_data)
        current_mse = encoder.calculate_mse(data, decoded_data)
        print(f"Current MSE: {current_mse} at interface size: {current_size}")

        if current_mse <= max_error:
            print("Desired MSE reached. Stopping training.")
            break
        
        if incremental_search:
            current_size += step_size
        else:
            current_size -= step_size

        if current_size >= data.shape[1] and incremental_search:
            print("Maximum interface size reached. Stopping training.")
            break

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

    debug_info = {}
    encoder_name = config.get('encoder_plugin', 'default_encoder')
    decoder_name = config.get('decoder_plugin', 'default_decoder')
    Encoder, encoder_params, Decoder, decoder_params = load_encoder_decoder_plugins(encoder_name, decoder_name)

    encoder = Encoder()
    decoder = Decoder()

    for column in data.columns:
        print(f"Processing column: {column}")
        column_data = data[[column]].values
        windowed_data = sliding_window(column_data, config['window_size'])

        trained_encoder, trained_decoder = train_autoencoder(
            encoder, decoder, windowed_data, config['mse_threshold'],
            config['initial_encoding_dim'], config['encoding_step_size'], config['incremental_search']
        )

        model_filename = f"{config['save_encoder']}_{column}.h5"
        trained_encoder.save(model_filename)
        print(f"Model saved as {model_filename}")

        decoded_data = trained_decoder.decode(trained_encoder.encode(windowed_data))
        mse = encoder.calculate_mse(windowed_data, decoded_data)
        print(f"Mean Squared Error for column {column}: {mse}")
        debug_info[f'mean_squared_error_{column}'] = mse

        trained_encoder.add_debug_info(debug_info)

        output_filename = f"{config['csv_output_path']}_{column}.csv"
        write_csv(output_filename, decoded_data, include_date=config['force_date'], headers=config['headers'])
        print(f"Output written to {output_filename}")

        if config['remote_log']:
            log_response = requests.post(
                config['remote_log'],
                auth=(config['remote_username'], config['remote_password']),
                json=debug_info
            )
            print(f"Remote log response: {log_response.text}")

    return decoded_data, debug_info
