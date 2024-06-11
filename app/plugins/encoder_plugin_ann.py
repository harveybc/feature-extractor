import numpy as np
from keras.models import Sequential, load_model, Model, save_model
from keras.layers import Dense, Input
from keras.optimizers import Adam

class Plugin:
    """
    An encoder plugin using a simple neural network based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'input_dim': None,
        'encoding_dim': None,
        'epochs': 10,
        'batch_size': 256
    }

    plugin_debug_vars = ['input_dim', 'encoding_dim', 'epochs', 'batch_size']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None
        self.encoder_model = None

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

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='tanh')(encoded)

        self.model = Model(inputs=input_layer, outputs=decoded)
        self.encoder_model = Model(inputs=self.model.input, outputs=encoded)
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, data):
        self.model.fit(data, data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)

    def encode(self, data):
        return self.encoder_model.predict(data)

    def save(self, file_path):
        save_model(self.model, file_path)

    def load(self, file_path):
        self.model = load_model(file_path)
        self.encoder_model = Model(inputs=self.model.input, outputs=self.model.layers[1].output)

    def calculate_mse(self, original_data, reconstructed_data):
        if original_data.shape != reconstructed_data.shape:
            reconstructed_data = reconstructed_data.reshape(original_data.shape)
        return np.mean(np.square(original_data - reconstructed_data))
