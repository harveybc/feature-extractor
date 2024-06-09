import numpy as np
from app.data_handler import load_csv, write_csv
from app.plugin_loader import load_encoder_decoder_plugins
import requests

def train_autoencoder(encoder, decoder, data, low_level_mse_threshold, max_epochs):
    current_mse = float('inf')
    epoch = 0

    while current_mse > low_level_mse_threshold and epoch < max_epochs:
        print(f"Training epoch {epoch + 1}/{max_epochs}")
        encoder.train(data)
        encoded_data = encoder.encode(data)
        decoder.train(encoded_data, data)
        decoded_data = decoder.decode(encoded_data)
        current_mse = encoder.calculate_mse(data, decoded_data)
        epoch += 1
        print(f"Current MSE: {current_mse}")

    return encoder, decoder, current_mse

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

        encoder_name = config.get('encoder_plugin', 'default')
        decoder_name = config.get('decoder_plugin', 'default')
        Encoder, encoder_params, Decoder, decoder_params = load_encoder_decoder_plugins(encoder_name, decoder_name)

        encoder = Encoder()
        decoder = Decoder()

        current_size = config['initial_encoding_dim']
        while (current_size > 0 and not config['incremental_search']) or (current_size <= config['window_size'] and config['incremental_search']):
            print(f"Configuring encoder and decoder with interface size {current_size}...")
            encoder.configure_size(input_dim=column_data.shape[1], encoding_dim=current_size)
            decoder.configure_size(encoding_dim=current_size, output_dim=column_data.shape[1])

            print(f"Training autoencoder with interface size {current_size}...")
            trained_encoder, trained_decoder, current_mse = train_autoencoder(
                encoder, decoder, column_data, config['mse_threshold'], config['epochs']
            )

            print(f"Current MSE: {current_mse} at interface size: {current_size}")
            if (config['incremental_search'] and current_mse <= config['mse_threshold']) or (not config['incremental_search'] and current_mse > config['mse_threshold']):
                print("Desired MSE reached or exceeded. Stopping optimization.")
                break

            if config['incremental_search']:
                current_size += config['encoding_step_size']
            else:
                current_size -= config['encoding_step_size']

        encoder_filename = f"{config['save_encoder']}_{col_index}.h5" if col_index > 0 else config['save_encoder']
        decoder_filename = f"{config['save_decoder']}_{col_index}.h5" if col_index > 0 else config['save_decoder']

        print(f"Saving encoder model to {encoder_filename}...")
        trained_encoder.save(encoder_filename)
        print(f"Saving decoder model to {decoder_filename}...")
        trained_decoder.save(decoder_filename)

        encoded_data = trained_encoder.encode(column_data)
        decoded_data = trained_decoder.decode(encoded_data)

        mse = trained_encoder.calculate_mse(column_data, decoded_data)
        print(f"Mean Squared Error for column {col_index}: {mse}")
        debug_info[f'mean_squared_error_{col_index}'] = mse

        trained_encoder.add_debug_info(debug_info)
        decoded_data_all.append(decoded_data)

        output_filename = f"{config['csv_output_path']}_{col_index}.csv"
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
