import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense
from keras.optimizers import Adam

class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin, input_dim, encoding_dim):
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder_model = None

    def build_autoencoder(self):
        encoder_input = Input(shape=(self.input_dim,), name="encoder_input")
        encoder_output = self.encoder_plugin.configure_size(self.input_dim, self.encoding_dim)
        decoder_output = self.decoder_plugin.configure_size(self.encoding_dim, self.input_dim)

        self.autoencoder_model = Model(inputs=encoder_input, outputs=decoder_output, name="autoencoder")
        self.autoencoder_model.compile(optimizer=Adam(), loss='mean_squared_error')

        print("Full Autoencoder Model Summary:")
        self.autoencoder_model.summary()

    def train_autoencoder(self, data, epochs, batch_size):
        print(f"Training autoencoder with data shape: {data.shape}")
        self.autoencoder_model.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=1)
        print("Training completed.")

    def encode(self, data):
        return self.encoder_plugin.encode(data)

    def decode(self, encoded_data):
        return self.decoder_plugin.decode(encoded_data)

    def save(self, encoder_path, decoder_path):
        self.encoder_plugin.save(encoder_path)
        self.decoder_plugin.save(decoder_path)

    def load(self, encoder_path, decoder_path):
        self.encoder_plugin.load(encoder_path)
        self.decoder_plugin.load(decoder_path)

    def calculate_mse(self, original_data, reconstructed_data):
        return np.mean(np.square(original_data - reconstructed_data))

# Debugging usage example
if __name__ == "__main__":
    from app.plugins.encoder_plugin_ann import Plugin as EncoderPlugin
    from app.plugins.decoder_plugin_ann import Plugin as DecoderPlugin

    encoder = EncoderPlugin()
    decoder = DecoderPlugin()

    autoencoder_manager = AutoencoderManager(encoder, decoder, input_dim=128, encoding_dim=4)
    autoencoder_manager.build_autoencoder()

    # Dummy data for testing
    data = np.random.random((1000, 128))
    autoencoder_manager.train_autoencoder(data, epochs=10, batch_size=256)