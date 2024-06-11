import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input
from keras.optimizers import Adam

class Plugin:
    """
    An encoder plugin using a simple neural network based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256
    }

    plugin_debug_vars = ['epochs', 'batch_size']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None
        self.autoencoder_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, input_dim, encoding_dim):
        self.params['input_dim'] = input_dim
        self.params['encoding_dim'] = encoding_dim

        # Encoder
        encoder_input = Input(shape=(input_dim,), name="encoder_input")
        encoder_output = Dense(encoding_dim, activation='relu', name="encoder_output")(encoder_input)
        self.encoder_model = Model(inputs=encoder_input, outputs=encoder_output, name="encoder")

        # Autoencoder
        decoder_output = Dense(input_dim, activation='tanh', name="decoder_output")(encoder_output)
        self.autoencoder_model = Model(inputs=encoder_input, outputs=decoder_output, name="autoencoder")
        self.autoencoder_model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Debugging messages to trace the model configuration
        print("Encoder Model Summary:")
        self.encoder_model.summary()
        print("Full Autoencoder Model Summary:")
        self.autoencoder_model.summary()

    def train(self, data):
        print(f"Training autoencoder with data shape: {data.shape}")
        self.autoencoder_model.fit(data, data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)
        print("Training completed.")

    def encode(self, data):
        print(f"Encoding data with shape: {data.shape}")
        encoded_data = self.encoder_model.predict(data)
        print(f"Encoded data shape: {encoded_data.shape}")
        return encoded_data

    def save(self, file_path):
        save_model(self.encoder_model, file_path)
        print(f"Encoder model saved to {file_path}")

    def load(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"Encoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], -
