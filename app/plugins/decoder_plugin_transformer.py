import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Flatten, Reshape, LayerNormalization, Dropout, Add, MultiHeadAttention
from keras.optimizers import Adam

class Plugin:
    """
    A decoder plugin using transformer layers.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256,
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'embedding_dim': 64,
        'num_heads': 8,
        'ff_dim_divisor': 2,
        'dropout_rate': 0.1
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'interface_size', 'output_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
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
        current_location = output_shape
        int_layers = 0
        while (current_size > interface_size) and (int_layers < (self.params['intermediate_layers'] + 1)):
            layers.append(current_location)
            current_size = max(current_size // layer_size_divisor, interface_size)
            current_location = interface_size + current_size
            int_layers += 1
        layers.append(interface_size)
        layers.reverse()

        print(f"Decoder Layer sizes: {layers}")

        # set input layer
        inputs = Input(shape=(interface_size,))
        x = Dense(layers[0], activation='relu')(inputs)
        x = Reshape((layers[0], 1))(x)

        for size in layers[1:]:
            embedding_dim = self.params['embedding_dim']
            num_heads = self.params['num_heads']
            ff_dim = size // self.params['ff_dim_divisor']
            dropout_rate = self.params['dropout_rate']

            x = Dense(embedding_dim)(x)
            attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
            attn_output = Dropout(dropout_rate)(attn_output)
            out1 = Add()([x, attn_output])
            out1 = LayerNormalization(epsilon=1e-6)(out1)

            ffn_output = Dense(ff_dim, activation='relu')(out1)
            ffn_output = Dense(embedding_dim)(ffn_output)
            ffn_output = Dropout(dropout_rate)(ffn_output)
            x = Add()([out1, ffn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

        x = Flatten()(x)
        outputs = Dense(output_shape)(x)

        self.model = Model(inputs=inputs, outputs=outputs, name="decoder")
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, encoded_data, original_data):
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)

    def decode(self, encoded_data):
        decoded_data = self.model.predict(encoded_data)
        return decoded_data

    def save(self, file_path):
        save_model(self.model, file_path)
        print(f"Decoder model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Decoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        mse = np.mean(np.square(original_data - reconstructed_data))
        return mse

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=4, output_shape=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
