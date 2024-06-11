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

        # Debugging message
        print(f"Configuring size with input_dim: {input_dim} and encoding_dim: {encoding_dim}")

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='tanh')(encoded)

        self.model = Model(inputs=input_layer, outputs=decoded)
        self.encoder_model = Model(inputs=input_layer, outputs=encoded)
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Debugging messages to trace the model configuration
        print(f"Encoder Model Summary: \n{self.encoder_model.summary()}")
        print(f"Full Model Summary: \n{self.model.summary()}")

    def train(self, data):
        # Debugging message
        print("Starting training...")
        self.model.fit(data, data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)
        print("Training completed.")

    def encode(self, data):
        # Debugging message
        print(f"Encoding data with shape: {data.shape}")
        encoded_data = self.encoder_model.predict(data)
        print(f"Encoded data shape: {encoded_data.shape}")
        return encoded_data

    def save(self, file_path):
        save_model(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        self.encoder_model = Model(inputs=self.model.input, outputs=self.model.layers[1].output)
        print(f"Model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        if original_data.shape != reconstructed_data.shape:
            reconstructed_data = reconstructed_data.reshape(original_data.shape)
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"Calculated MSE: {mse}")
        return mse

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(input_dim=128, encoding_dim=4)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
