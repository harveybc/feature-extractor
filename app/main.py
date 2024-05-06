import sys
from app.cli import parse_args
from app.config import MODEL_SAVE_PATH, MODEL_LOAD_PATH, OUTPUT_PATH, MINIMUM_MSE_THRESHOLD
from app.data_handler import load_csv
import pkg_resources

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

def train_autoencoder(encoder, decoder, data, max_mse):
    """
    Train an autoencoder until the maximum MSE threshold is reached.
    
    Args:
        encoder (object): Encoder plugin instance.
        decoder (object): Decoder plugin instance.
        data (DataFrame): Input data.
        max_mse (float): Maximum allowed MSE for stopping the training.
    
    Returns:
        tuple: The trained encoder and decoder.
    """
    current_mse = float('inf')
    while current_mse > max_mse:
        encoder.train(data)
        encoded_data = encoder.encode(data)
        decoder.train(encoded_data)
        decoded_data = decoder.decode(encoded_data)
        current_mse = decoder.calculate_mse(data, decoded_data)
        print(f"Current MSE: {current_mse}")
        if current_mse <= max_mse:
            print("Desired MSE reached. Stopping training.")
            break
    return encoder, decoder

def main():
    # Parse command line arguments
    args = parse_args()

    # Load the CSV data
    data = load_csv(args.csv_file)

    # Initialize encoder and decoder
    Encoder = load_plugin('feature_extractor.encoders', args.encoder_plugin if args.encoder_plugin else 'default_encoder')
    Decoder = load_plugin('feature_extractor.decoders', args.decoder_plugin if args.decoder_plugin else 'default_decoder')
    encoder = Encoder()
    decoder = Decoder()

    # Perform operations based on the arguments
    if args.save_encoder:
        trained_encoder, trained_decoder = train_autoencoder(encoder, decoder, data, args.min_mse if args.min_mse else MINIMUM_MSE_THRESHOLD)
        trained_encoder.save(MODEL_SAVE_PATH + args.save_encoder)
        trained_decoder.save(MODEL_SAVE_PATH + args.save_encoder.replace('encoder', 'decoder'))

    if args.load_encoder_params:
        encoder.load(MODEL_LOAD_PATH + args.load_encoder_params)

    if args.load_decoder_params:
        decoder.load(MODEL_LOAD_PATH + args.load_decoder_params)

    if args.evaluate_encoder or args.evaluate_decoder:
        if not hasattr(encoder, 'load') or not hasattr(decoder, 'load'):
            print("Error: Encoder or Decoder does not support load operation.")
            sys.exit(1)
        encoder.load(MODEL_LOAD_PATH + args.load_encoder_params)
        decoder.load(MODEL_LOAD_PATH + args.load_decoder_params)
        if args.evaluate_encoder:
            encoded_data = encoder.encode(data)
            with open(OUTPUT_PATH + args.evaluate_encoder, 'w') as f:
                f.write(str(encoded_data))
        if args.evaluate_decoder:
            encoded_data = encoder.encode(data)
            decoded_data = decoder.decode(encoded_data)
            with open(OUTPUT_PATH + args.evaluate_decoder, 'w') as f:
                f.write(str(decoded_data))

if __name__ == '__main__':
    main()
