import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from keras.optimizers import Adam

class Plugin:
    """
    An encoder plugin using a convolutional neural network (CNN) based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256,
        'intermediate_layers': 1
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, input_shape, interface_size):
        self.params['input_shape'] = input_shape

        layers = []
        current_size = input_shape
        layer_size_divisor = 1 + self.params['intermediate_layers']
        while current_size > interface_size:
            layers.append((current_size, interface_size))
            current_size = max(current_size // layer_size_divisor, interface_size)
        
        # Debugging message
        print(f"Decoder Layer sizes: {layers}")

        inputs = Input(shape=(input_shape, 1))
        x = inputs
        for size, pool_size in layers:
            x = Conv1D(filters=size, kernel_size=3, activation='relu', padding='same')(x)
            x = MaxPooling1D(pool_size=pool_size)(x)

        x = Flatten()(x)
        x = Dense(layers[-1][0], activation='relu')(x)
        outputs = Dense(interface_size)(x)

        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder")
        self.encoder_model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Debugging messages to trace the model configuration
        print("Encoder Model Summary:")
        self.encoder_model.summary()

    def train(self, data):
        print(f"Training encoder with data shape: {data.shape}")
        self.encoder_model.fit(data, data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)
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

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(input_shape=128, interface_size=4)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
