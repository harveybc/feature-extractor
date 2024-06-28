# encoder_plugin_transformer.py

import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, LayerNormalization, Dropout
from keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Add
from keras.optimizers import Adam

class Plugin:
    """
    An encoder plugin using a transformer-based neural network, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256,
        'num_heads': 8,
        'ff_dim': 64,
        'num_layers': 1,
        'dropout_rate': 0.1
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_shape', 'num_heads', 'ff_dim', 'num_layers', 'dropout_rate']

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

        # Transformer Encoder Layer
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
            x = Dropout(dropout)(x)
            x = Add()([x, inputs])
            x = LayerNormalization(epsilon=1e-6)(x)
            res = x
            x = Dense(ff_dim, activation="relu")(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            x = Add()([x, res])
            return LayerNormalization(epsilon=1e-6)(x)

        inputs = Input(shape=(input_shape, 1))
        x = inputs

        for _ in range(self.params['num_layers']):
            x = transformer_encoder(x, self.params['ff_dim'], self.params['num_heads'], self.params['ff_dim'], self.params['dropout_rate'])

        x = GlobalAveragePooling1D()(x)
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
