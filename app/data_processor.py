import sys
import requests
import numpy as np
from app.data_handler import load_csv, write_csv, sliding_window
from app.plugin_loader import load_encoder_decoder_plugins

def train_autoencoder(encoder, decoder, data, max_error, initial_size, step_size):
    current_size = initial_size
    current_mse = float('inf')

    while current_size > 0 and current_mse > max_error:
        print(f"Configuring encoder and decoder with interface size {current_size}...")
        encoder.configure_size(input_dim=data.shape[1], encoding_dim=current_size)
        decoder.configure_size(encoding_dim=current_size, output_dim=data.shape[1])

        print(f"Training encoder with interface size {current_size}...")
        encoder.train(data)

        encoded_data = encoder.encode(data)

        print(f"Training decoder with interface size {current_size}...")
        decoder.train(encoded_data, data)

        decoded_data = decoder.decode(encoded_data)
        current_mse = encoder.calculate_mse(data, decoded_data)
        print(f"Current MSE: {current_mse} at interface size: {current_size}")

        if current_mse <= max_error:
            print("Desired MSE reached. Stopping training.")
            break

        current_size -= step_size

    return encoder, decoder

def process_data(config):
    print(f"Loading data from {config['csv_file']}...")
    data = load_csv(config['csv_file'], headers=config['headers'])
    print(f"Data loaded: {len(data)} rows and {data.shape[1]} columns.")
    
    # Ensure all data is numerical
    print("Converting data to float...")
    data = data.apply(pd.to_numeric, errors='coerce')
    print(f"Data types:\n{data.dtypes}")

    print("Creating windowed data...")
    windowed_data = sliding_window(data, config['window_size'])
    print(f"Data loaded and windowed. Number of windows: {len(windowed_data)}.")

    print("Using encoder plugin:", config['encoder_plugin'])
    print("Using decoder plugin:", config['decoder_plugin'])

    Encoder, encoder_params, Decoder, decoder_params = load_encoder_decoder_plugins(config['encoder_plugin'], config['decoder_plugin'])

    encoder = Encoder()
    decoder = Decoder()

    debug_info = {}

    for index, series_data in enumerate(windowed_data):
        print(f"Training autoencoder for window {index}...")
        
        # Ensure all data is numerical for each window
        series_data = np.asarray(series_data).astype(np.float32)
        print(f"Window data types:\n{series_data.dtype}")

        trained_encoder, trained_decoder = train_autoencoder(encoder, decoder, series_data, config['max_error'], config['initial_size'], config['step_size'])

        model_filename = f"{config['save_encoder']}_{index}.h5"
        print(f"Saving encoder model to {model_filename}...")
        trained_encoder.save(model_filename)

        model_filename = f"{config['save_decoder']}_{index}.h5"
        print(f"Saving decoder model to {model_filename}...")
        trained_decoder.save(model_filename)

        encoded_data = trained_encoder.encode(series_data)
        decoded_data = trained_decoder.decode(encoded_data)

        mse = encoder.calculate_mse(series_data, decoded_data)
        print(f"Mean Squared Error for window {index}: {mse}")
        debug_info[f'mean_squared_error_{index}'] = mse

        trained_encoder.add_debug_info(debug_info)
        trained_decoder.add_debug_info(debug_info)

        output_filename = f"{config['csv_output_path']}_{index}.csv"
        print(f"Writing decoded data to {output_filename}...")
        write_csv(output_filename, decoded_data, include_date=config['force_date'], headers=config['headers'])

        if config['remote_log']:
            log_response = requests.post(
                config['remote_log'],
                auth=(config['remote_username'], config['remote_password']),
                json=debug_info
            )
            print(f"Remote log response: {log_response.text}")

    return decoded_data, debug_info
