import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling1D, LayerNormalization, Dropout, Add, Activation
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention

class Plugin:
    """
    An encoder plugin using transformer layers.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256,
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'ff_dim_divisor': 2,
        'dropout_rate': 0.1
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

        # add transformer layers
        for size in layers:
            ff_dim = size // self.params['ff_dim_divisor']
            if size < 64:
                num_heads = 2
            elif 64 <= size < 128:
                num_heads = 4
            else:
                num_heads = 8

            dropout_rate = self.params['dropout_rate']
            
            x = Dense(size)(x)
            x = MultiHeadAttention(head_num=num_heads)(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Dropout(dropout_rate)(x)
            
            ffn_output = Dense(ff_dim, activation='relu')(x)
            ffn_output = Dense(size)(ffn_output)
            ffn_output = Dropout(dropout_rate)(ffn_output)
            x = Add()([x, ffn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

        x = GlobalAveragePooling1D()(x)
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
