import sys
import requests
from app.data_handler import load_csv, write_csv, sliding_window
from app.plugin_loader import load_plugin

def train_autoencoder(encoder, data, max_error, initial_size, step_size):
    """
    Train an autoencoder with decreasing interface size until the max_error is reached.

    Args:
        encoder (object): The encoder plugin instance.
        data (np.array): The data to train on.
        max_error (float): Maximum mean squared error threshold.
        initial_size (int): Initial size of the encoder/decoder interface.
        step_size (int): Step size to reduce the interface size.

    Returns:
        object: Trained encoder instance.
    """
    current_size = initial_size
    current_mse = float('inf')
    while current_size > 0 and current_mse > max_error:
        encoder.configure_size(input_dim=data.shape[1], encoding_dim=current_size)
        encoder.train(data)
        encoded_data = encoder.encode(data)
        decoded_data = encoder.model.predict(encoded_data)
        current_mse = encoder.calculate_mse(data, decoded_data)
        print(f"Current MSE: {current_mse} at interface size: {current_size}")
        if current_mse <= max_error:
            print("Desired MSE reached. Stopping training.")
            break
        current_size -= step_size
        response = requests.post(
            'http://localhost:60500/feature_extractor/fe_training_error',
            auth=('test', 'pass'),
            data={'mse': current_mse, 'interface_size': current_size, 'config_id': 1}
        )
        print(f"Response from data-logger: {response.text}")
    return encoder

def process_data(config):
    """
    Process the data using the specified encoder plugin.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: Processed data and debug information.
    """
    data = load_csv(config['csv_file'], headers=config['headers'])
    windowed_data = sliding_window(data, config['window_size'])

    print("Loaded data:\n", data.head())

    debug_info = {}

    for index, series_data in enumerate(windowed_data):
        Encoder = load_plugin('feature_extractor.encoders', config.get('encoder_plugin', 'default_encoder'))
        encoder = Encoder()
        trained_encoder = train_autoencoder(encoder, series_data, config['max_error'], config['initial_size'], config['step_size'])

        model_filename = f"{config['save_encoder']}_{index}.h5"
        trained_encoder.save(model_filename)
        print(f"Model saved as {model_filename}")

        encoded_data = trained_encoder.encode(series_data)
        decoded_data = trained_encoder.model.predict(encoded_data)

        mse = encoder.calculate_mse(series_data, decoded_data)
        print(f"Mean Squared Error: {mse}")
        debug_info[f'mean_squared_error_{index}'] = mse

        trained_encoder.add_debug_info(debug_info)

        output_filename = f"{config['output_file']}_{index}.csv"
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
