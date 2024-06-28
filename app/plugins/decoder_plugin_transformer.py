# decoder_plugin_transformer.py

import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, LayerNormalization, Dropout, Reshape, Flatten, Add, MultiHeadAttention
from keras.optimizers import Adam

class Plugin:
    """
    A decoder plugin using a transformer-based neural network, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256,
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'num_heads': 8,
        'ff_dim_divisor': 2,
        'dropout_rate': 0.1
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'interface_size', 'output_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.decoder_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, interface_size, output_shape):
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        layers = []
        current_size = output_shape
        layer_size_divisor = self.params['layer_size_divisor']
        int_layers = 0
        while (current_size > interface_size) and (int_layers < (self.params['intermediate_layers'] + 1)):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, interface_size)
            int_layers += 1
        layers.append(interface_size)
        layers.reverse()

        # Debugging message
        print(f"Decoder Layer sizes: {layers}")

        # set input layer
        inputs = Input(shape=(interface_size,))
        x = inputs
        x = Reshape((interface_size, 1))(x)

        # add transformer layers, calculating their sizes based on the layer's size
        for size in layers:
            # Attention Layer
            attn_output = MultiHeadAttention(num_heads=self.params['num_heads'], key_dim=size)(x, x)
            attn_output = Dropout(self.params['dropout_rate'])(attn_output)
            x = Add()([x, attn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

            # Feed Forward Network
            ffn_dim = size // self.params['ff_dim_divisor']
            ffn_output = Dense(ffn_dim, activation="relu")(x)
            ffn_output = Dropout(self.params['dropout_rate'])(ffn_output)
            ffn_output = Dense(size)(ffn_output)
            x = Add()([x, ffn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

        x = Flatten()(x)
        outputs = Dense(output_shape, activation='tanh')(x)

        self.decoder_model = Model(inputs=inputs, outputs=outputs, name="decoder")
        self.decoder_model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, encoded_data, original_data):
        print(f"Training decoder with data shape: {encoded_data.shape}")
        self.decoder_model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)
        print("Training completed.")

    def decode(self, encoded_data):
        print(f"Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.decoder_model.predict(encoded_data)
        print(f"Decoded data shape: {decoded_data.shape}")
        return decoded_data

    def save(self, file_path):
        save_model(self.decoder_model, file_path)
        print(f"Decoder model saved to {file_path}")

    def load(self, file_path):
        self.decoder_model = load_model(file_path)
        print(f"Decoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], -1))
        reconstructed_data = reconstructed_data.reshape((reconstructed_data.shape[0], -1))
        mse = np.mean(np.square(original_data - reconstructed_data))
        return mse

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=4, output_shape=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
