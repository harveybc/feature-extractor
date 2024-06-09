import sys
import requests
from app.data_handler import load_csv, write_csv, sliding_window
from app.plugin_loader import load_encoder_decoder_plugins

def train_autoencoder(encoder, decoder, data, max_error, initial_size, step_size):
    """
    Train an autoencoder with decreasing interface size until the max_error is reached.

    Args:
        encoder (object): The encoder plugin instance.
        decoder (object): The decoder plugin instance.
        data (np.array): The data to train on.
        max_error (float): Maximum mean squared error threshold.
        initial_size (int): Initial size of the encoder/decoder interface.
        step_size (int): Step size to reduce the interface size.

    Returns:
        tuple: Trained encoder and decoder instances.
    """
    current_size = initial_size
    current_mse = float('inf')
    while current_size > 0 and current_mse > max_error:
        print(f"Configuring encoder and decoder with interface size: {current_size}")
        encoder.configure_size(input_dim=data.shape[1], encoding_dim=current_size)
        decoder.configure_size(encoding_dim=current_size, output_dim=data.shape[1])
        
        print(f"Training encoder with interface size: {current_size}")
        encoder.train(data)
        encoded_data = encoder.encode(data)
        
        print(f"Training decoder with interface size: {current_size}")
        decoder.train(encoded_data, data)
        decoded_data = decoder.decode(encoded_data)
        
        current_mse = encoder.calculate_mse(data, decoded_data)
        print(f"Current MSE: {current_mse} at interface size: {current_size}")
        
        if current_mse <= max_error:
            print("Desired MSE reached. Stopping training.")
            break
        
        current_size -= step_size

        # Logging training error to remote server (if needed)
        if config['remote_log']:
            response = requests.post(
                config['remote_log'],
                auth=(config['remote_username'], config['remote_password']),
                json={'mse': current_mse, 'interface_size': current_size, 'config_id': 1}
            )
            print(f"Response from data-logger: {response.text}")

    return encoder, decoder

def process_data(config):
    """
    Process the data using the specified encoder and decoder plugins.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: Processed data and debug information.
    """
    print(f"Loading data from {config['csv_file']}...")
    data = load_csv(config['csv_file'], headers=config['headers'])
    windowed_data = sliding_window(config['csv_file'], config['window_size'], data)
    print("Data loaded and windowed.")

    debug_info = {}
    decoded_data = None

    encoder_name = config.get('encoder_plugin', 'default')
    decoder_name = config.get('decoder_plugin', 'default')
    Encoder, encoder_params, Decoder, decoder_params = load_encoder_decoder_plugins(encoder_name, decoder_name)

    print(f"Using encoder plugin: {encoder_name}")
    print(f"Using decoder plugin: {decoder_name}")

    encoder = Encoder()
    decoder = Decoder()

    for index, series_data in enumerate(windowed_data):
        print(f"Processing windowed data index: {index}")
        trained_encoder, trained_decoder = train_autoencoder(
            encoder, decoder, series_data, config['max_error'], config['initial_size'], config['step_size']
        )

        model_filename = f"{config['save_encoder']}_{index}.h5"
        print(f"Saving encoder model to {model_filename}...")
        trained_encoder.save(model_filename)

        model_filename = f"{config['save_decoder']}_{index}.h5"
        print(f"Saving decoder model to {model_filename}...")
        trained_decoder.save(model_filename)

        encoded_data = trained_encoder.encode(series_data)
        decoded_data = trained_decoder.decode(encoded_data)

        mse = trained_encoder.calculate_mse(series_data, decoded_data)
        print(f"Mean Squared Error for window {index}: {mse}")
        debug_info[f'mean_squared_error_{index}'] = mse

        trained_encoder.add_debug_info(debug_info)
        trained_decoder.add_debug_info(debug_info)

        output_filename = f"{config['output_file']}_{index}.csv"
        print(f"Writing decoded data to {output_filename}...")
        write_csv(output_filename, decoded_data, include_date=config['force_date'], headers=config['headers'])

        if config['remote_log']:
            log_response = requests.post(
                config['remote_log'],
                auth=(config['remote_username'], config['remote_password']),
                json=debug_info
            )
            print(f"Remote log response: {log_response.text}")

    if decoded_data is None:
        print("No data processed. Check your configuration and input data.")
        sys.exit(1)

    return decoded_data, debug_info
