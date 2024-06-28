import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Bidirectional, Dense, Input, Flatten
from keras.optimizers import Adam

class Plugin:
    """
    An encoder plugin using a Bidirectional Long Short-Term Memory (Bi-LSTM) network based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256,
        'intermediate_layers': 1,
        'layer_size_divisor': 2
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
        layer_size_divisor = self.params['layer_size_divisor'] 
        current_location = input_shape
        int_layers = 0
        while (current_size > interface_size) and (int_layers < (self.params['intermediate_layers']+1)):
            layers.append(current_location)
            current_size = max(current_size // layer_size_divisor, interface_size)
            current_location = interface_size + current_size
            int_layers += 1
        layers.append(interface_size)
        # Debugging message
        print(f"Encoder Layer sizes: {layers}")

        # set input layer
        inputs = Input(shape=(input_shape, 1))
        x = inputs

        # add Bi-LSTM layers
        layers_index = 0
        for size in layers:
            layers_index += 1

            # add the Bi-LSTM layers
            if layers_index == 1:
                x = Bidirectional(LSTM(units=size, activation='tanh', return_sequences=True))(x)
            else:
                x = Bidirectional(LSTM(units=size, activation='tanh', return_sequences=(layers_index < len(layers))))(x)
        
        x = Flatten()(x)
        outputs = Dense(interface_size)(x)
        
        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder")
        self.encoder_model.compile(optimizer=Adam(), loss='mean_squared_error')

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
