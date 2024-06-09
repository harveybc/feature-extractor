import numpy as np
from app.data_handler import load_csv, write_csv, sliding_window
from app.plugin_loader import load_encoder_decoder_plugins
import requests

def train_autoencoder(encoder, decoder, data, mse_threshold, initial_size, step_size, incremental_search):
    current_size = initial_size
    current_mse = float('inf')
    while current_size > 0 and current_mse > mse_threshold and current_size < data.shape[1]:
        print(f"Configuring encoder and decoder with interface size {current_size}...")
        encoder.configure_size(input_dim=data.shape[1], encoding_dim=current_size)
        decoder.configure_size(encoding_dim=current_size, output_dim=data.shape[1])

        print(f"Training encoder with interface size {current_size}...")
        encoder.train(data)

        encoded_data = encoder.encode(data)
        print("Training decoder...")
        decoder.train(encoded_data, data)

        decoded_data = decoder.decode(encoded_data)
        current_mse = encoder.calculate_mse(data, decoded_data)
        print(f"Current MSE: {current_mse} at interface size: {current_size}")

        if current_mse <= mse_threshold:
            print("Desired MSE reached. Stopping training.")
            break

        if incremental_search:
            current_size += step_size
        else:
            current_size -= step_size

        response = requests.post(
            'http://localhost:60500/feature_extractor/fe_training_error',
            auth=('test', 'pass'),
            data={'mse': current_mse, 'interface_size': current_size, 'config_id': 1}
        )
        print(f"Response from data-logger: {response.text}")

    return encoder, decoder

def process_data(config):
    print("Loading data from", config['csv_file'])
    data = load_csv(config['csv_file'], headers=config['headers'])
    print(f"Data loaded: {len(data)} rows and {data.shape[1]} columns.")

    print("Converting data to float...")
    data = data.astype(float)
    print("Data types:\n", data.dtypes)

    debug_info = {}
    decoded_data_all = []

    for col_index in range(data.shape[1]):
        print(f"Processing column: {col_index}")
        column_data = data.iloc[:, col_index].values.reshape(-1, 1)
        
        if config['window_size'] > len(column_data):
            print(f"Warning: Window size {config['window_size']} is larger than the data length {len(column_data)}. Adjusting window size to {len(column_data)}.")
            window_size = len(column_data)
        else:
            window_size = config['window_size']
            
        windowed_data = sliding_window(column_data, window_size)
        print(f"Windowed data shape: {windowed_data.shape}")

        encoder_name = config.get('encoder_plugin', 'default')
        decoder_name = config.get('decoder_plugin', 'default')
        Encoder, encoder_params, Decoder, decoder_params = load_encoder_decoder_plugins(encoder_name, decoder_name)

        encoder = Encoder()
        decoder = Decoder()

        total_windows = len(windowed_data)
        for window_index, series_data in enumerate(windowed_data):
            print(f"Training autoencoder for window {window_index + 1}/{total_windows}...")
            trained_encoder, trained_decoder = train_autoencoder(
                encoder, decoder, series_data, config['mse_threshold'], config['initial_size'], config['step_size'], config['incremental_search']
            )

            encoder_filename = f"{config['save_encoder']}_{col_index}.h5" if col_index > 0 else config['save_encoder']
            decoder_filename = f"{config['save_decoder']}_{col_index}.h5" if col_index > 0 else config['save_decoder']

            print(f"Saving encoder model to {encoder_filename}...")
            trained_encoder.save(encoder_filename)
            print(f"Saving decoder model to {decoder_filename}...")
            trained_decoder.save(decoder_filename)

            encoded_data = trained_encoder.encode(series_data)
            decoded_data = trained_decoder.decode(encoded_data)

            mse = trained_encoder.calculate_mse(series_data, decoded_data)
            print(f"Mean Squared Error for window {window_index + 1}/{total_windows} on column {col_index}: {mse}")
            debug_info[f'mean_squared_error_{window_index}_{col_index}'] = mse

            trained_encoder.add_debug_info(debug_info)
            decoded_data_all.append(decoded_data)

            output_filename = f"{config['csv_output_path']}_{window_index}_{col_index}.csv"
            print(f"Writing decoded data to {output_filename}...")
            write_csv(output_filename, decoded_data, include_date=config['force_date'], headers=config['headers'])

            if config['remote_log']:
                log_response = requests.post(
                    config['remote_log'],
                    auth=(config['remote_username'], config['remote_password']),
                    json=debug_info
                )
                print(f"Remote log response: {log_response.text}")

    return np.concatenate(decoded_data_all, axis=0), debug_info
