import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam

class AutoencoderManager:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder_model = None
        self.encoder_model = None
        self.decoder_model = None

    def build_autoencoder(self):
        # Encoder
        encoder_input = Input(shape=(self.input_dim,), name="encoder_input")
        encoder_output = Dense(self.encoding_dim, activation='relu', name="encoder_output")(encoder_input)
        self.encoder_model = Model(inputs=encoder_input, outputs=encoder_output, name="encoder")

        # Decoder
        decoder_input = Input(shape=(self.encoding_dim,), name="decoder_input")
        decoder_output = Dense(self.input_dim, activation='tanh', name="decoder_output")(decoder_input)
        self.decoder_model = Model(inputs=decoder_input, outputs=decoder_output, name="decoder")

        # Autoencoder
        autoencoder_output = self.decoder_model(encoder_output)
        self.autoencoder_model = Model(inputs=encoder_input, outputs=autoencoder_output, name="autoencoder")
        self.autoencoder_model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Debugging messages to trace the model configuration
        print("Encoder Model Summary:")
        self.encoder_model.summary()
        print("Decoder Model Summary:")
        self.decoder_model.summary()
        print("Full Autoencoder Model Summary:")
        self.autoencoder_model.summary()

    def train_autoencoder(self, data, epochs=10, batch_size=256):
        print(f"Training autoencoder with data shape: {data.shape}")
        self.autoencoder_model.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=1)
        print("Training completed.")

    def encode_data(self, data):
        print(f"Encoding data with shape: {data.shape}")
        encoded_data = self.encoder_model.predict(data)
        print(f"Encoded data shape: {encoded_data.shape}")
        return encoded_data

    def decode_data(self, encoded_data):
        print(f"Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.decoder_model.predict(encoded_data)
        print(f"Decoded data shape: {decoded_data.shape}")
        return decoded_data

    def save_encoder(self, file_path):
        self.encoder_model.save(file_path)
        print(f"Encoder model saved to {file_path}")

    def save_decoder(self, file_path):
        self.decoder_model.save(file_path)
        print(f"Decoder model saved to {file_path}")

    def load_encoder(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"Encoder model loaded from {file_path}")

    def load_decoder(self, file_path):
        self.decoder_model = load_model(file_path)
        print(f"Decoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], -1))  # Flatten the data
        reconstructed_data = reconstructed_data.reshape((original_data.shape[0], -1))  # Flatten the data
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"Calculated MSE: {mse}")
        return mse
