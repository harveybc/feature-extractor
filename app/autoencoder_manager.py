import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

class AutoencoderManager:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder_model = None
        self.encoder_model = None

    def build_autoencoder(self):
        # Encoder
        encoder_input = Input(shape=(self.input_dim,), name="encoder_input")
        encoder_output = Dense(self.encoding_dim, activation='relu', name="encoder_output")(encoder_input)
        self.encoder_model = Model(inputs=encoder_input, outputs=encoder_output, name="encoder")

        # Decoder
        decoder_output = Dense(self.input_dim, activation='tanh', name="decoder_output")(encoder_output)
        self.autoencoder_model = Model(inputs=encoder_input, outputs=decoder_output, name="autoencoder")
        self.autoencoder_model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Debugging messages to trace the model configuration
        print("Encoder Model Summary:")
        self.encoder_model.summary()
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

    def save_encoder(self, file_path):
        self.encoder_model.save(file_path)
        print(f"Encoder model saved to {file_path}")

    def load_encoder(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"Encoder model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    autoencoder_manager = AutoencoderManager(input_dim=128, encoding_dim=4)
    autoencoder_manager.build_autoencoder()
    # Assuming data is already loaded and preprocessed
    # data = np.random.rand(8513, 128)  # Example data
    # autoencoder_manager.train_autoencoder(data)
    debug_info = {"input_dim": autoencoder_manager.input_dim, "encoding_dim": autoencoder_manager.encoding_dim}
    print(f"Debug Info: {debug_info}")
