import sys
from app.cli import parse_args
from app.config import MODEL_SAVE_PATH, MODEL_LOAD_PATH, OUTPUT_PATH, MINIMUM_MSE_THRESHOLD
from app.data_handler import load_csv
import pkg_resources
import requests

def load_plugin(plugin_group, plugin_name):
    """
    Load a plugin based on the group and name specified.
    """
    try:
        entry_point = next(pkg_resources.iter_entry_points(plugin_group, plugin_name))
        return entry_point.load()
    except StopIteration:
        print(f"Plugin {plugin_name} not found in group {plugin_group}.", file=sys.stderr)
        sys.exit(1)

def train_autoencoder(encoder, data, max_error, initial_size, step_size):
    """
    Train an autoencoder with decreasing interface size until the max_error is reached.
    """
    current_size = initial_size
    current_mse = float('inf')
    while current_size > 0 and current_mse > max_error:
        encoder.configure_size(current_size)
        encoder.train(data)
        encoded_data = encoder.encode(data)
        decoded_data = encoder.decode(encoded_data)
        current_mse = encoder.calculate_mse(data, decoded_data)
        print(f"Current MSE: {current_mse} at interface size: {current_size}")
        if current_mse <= max_error:
            print("Desired MSE reached. Stopping training.")
            break
        current_size -= step_size
        # Connect to data-logger on localhost:60500 via HTTP POST request with basic authentication using 
        # username 'test' and password 'pass', and send the current_mse and current_size as form data
        response = requests.post('http://localhost:60500/feature_extractor/fe_training_error', auth=('test', 'pass'), data={'mse': current_mse, 'interface_size': current_size, 'config_id': 1})
        print(f"Response from data-logger: {response.text}")
    return encoder

def main():
    # Parse command line arguments
    args = parse_args()

    # Load the CSV data
    data_series = load_csv(args.csv_file, args.window_size)

    # Train a separate autoencoder for each time series column
    for index, series_data in enumerate(data_series):
        Encoder = load_plugin('feature_extractor.encoders', args.encoder_plugin if args.encoder_plugin else 'default_encoder')
        encoder = Encoder()
        trained_encoder = train_autoencoder(encoder, series_data, args.max_error, args.initial_size, args.step_size)
        model_filename = f"{MODEL_SAVE_PATH}{args.save_encoder}_{index}.h5"
        trained_encoder.save(model_filename)
        print(f"Model saved as {model_filename}")

if __name__ == '__main__':
    main()
