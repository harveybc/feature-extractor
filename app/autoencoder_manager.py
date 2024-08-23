import numpy as np
from keras.models import Model, load_model
from keras.optimizers import Adam

class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin):
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.autoencoder_model = None
        self.encoder_model = None
        self.decoder_model = None
        print(f"[AutoencoderManager] Initialized with encoder plugin and decoder plugin")

    def build_autoencoder(self, input_shape, interface_size, config):
        try:
            print("[build_autoencoder] Starting to build autoencoder...")
            
            # Configure encoder and decoder sizes
            self.encoder_plugin.configure_size(input_shape, interface_size)
            self.decoder_plugin.configure_size(interface_size, input_shape)
            
            # Get the encoder model
            self.encoder_model = self.encoder_plugin.encoder_model
            print("[build_autoencoder] Encoder model built and compiled successfully")
            self.encoder_model.summary()

            # Get the decoder model
            self.decoder_model = self.decoder_plugin.model
            print("[build_autoencoder] Decoder model built and compiled successfully")
            self.decoder_model.summary()

            # Build autoencoder model
            autoencoder_output = self.decoder_model(self.encoder_model.output)
            self.autoencoder_model = Model(inputs=self.encoder_model.input, outputs=autoencoder_output, name="autoencoder")
            adam_optimizer = Adam(
                learning_rate= config['learning_rate'],   # Set the learning rate
                beta_1=0.9,            # Default value
                beta_2=0.999,          # Default value
                epsilon=1e-7,          # Default value
                amsgrad=False          # Default value
            )
            self.autoencoder_model.compile(optimizer=adam_optimizer, loss='mae')
            print("[build_autoencoder] Autoencoder model built and compiled successfully")
            self.autoencoder_model.summary()
        except Exception as e:
            print(f"[build_autoencoder] Exception occurred: {e}")
            raise

    def train_autoencoder(self, data, epochs=100, batch_size=32):
        try:
            if isinstance(data, tuple):
                data = data[0]
            print(f"[train_autoencoder] Training autoencoder with data shape: {data.shape}")
            self.autoencoder_model.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=1)
            print("[train_autoencoder] Training completed.")
        except Exception as e:
            print(f"[train_autoencoder] Exception occurred during training: {e}")
            raise

    def encode_data(self, data):
        print(f"[encode_data] Encoding data with shape: {data.shape}")
        encoded_data = self.encoder_model.predict(data)
        print(f"[encode_data] Encoded data shape: {encoded_data.shape}")
        return encoded_data

    def decode_data(self, encoded_data):
        print(f"[decode_data] Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.decoder_model.predict(encoded_data)
        print(f"[decode_data] Decoded data shape: {decoded_data.shape}")
        return decoded_data

    def save_encoder(self, file_path):
        self.encoder_model.save(file_path)
        print(f"[save_encoder] Encoder model saved to {file_path}")

    def save_decoder(self, file_path):
        self.decoder_model.save(file_path)
        print(f"[save_decoder] Decoder model saved to {file_path}")

    def load_encoder(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"[load_encoder] Encoder model loaded from {file_path}")

    def load_decoder(self, file_path):
        self.decoder_model = load_model(file_path)
        print(f"[load_decoder] Decoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        # print the shapes of the original data and the reconstructed_data
        print(f"[calculate_mse] Original data shape: {original_data.shape}")
        print(f"[calculate_mse] Reconstructed data shape: {reconstructed_data.shape}")
        #if isinstance(original_data, tuple):
        #    original_data = original_data[1] 
        #if isinstance(reconstructed_data, tuple):
        #    reconstructed_data = reconstructed_data[1]

        original_data = original_data.reshape((original_data.shape[0], -1))
        reconstructed_data = reconstructed_data.reshape((original_data.shape[0], -1))
        print(f"[calculate_mse] Original data shape after reshaping: {original_data.shape}")
        print(f"[calculate_mse] Reconstructed data shape after reshaping: {reconstructed_data.shape}")
        
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"[calculate_mse] Calculated MSE: {mse}")
        return mse

    def calculate_mae(self, original_data, reconstructed_data):
        # print the shapes of the original data and the reconstructed_data
        print(f"[calculate_mae] Original data shape: {original_data.shape}")
        print(f"[calculate_mae] Reconstructed data shape: {reconstructed_data.shape}")
        original_data = original_data.reshape((original_data.shape[0], -1))
        reconstructed_data = reconstructed_data.reshape((original_data.shape[0], -1))
        print(f"[calculate_mae] Original data shape after reshaping: {original_data.shape}")
        print(f"[calculate_mae] Reconstructed data shape after reshaping: {reconstructed_data.shape}")
        mae = np.mean(np.abs(original_data - reconstructed_data))
        print(f"[calculate_mae] Calculated MAE: {mae}")
        return mae
