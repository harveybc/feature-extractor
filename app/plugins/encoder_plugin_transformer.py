import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, GlobalAveragePooling1D, LayerNormalization, Dropout, Add, Flatten
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber

class Plugin:
    """
    A transformer-based encoder plugin.
    Configurable similarly to the ANN plugin.
    """

    plugin_params = {
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'ff_dim_divisor': 2,
        'learning_rate': 0.00001,
        'dropout_rate': 0.1,
        # Optionally, you can define an initial_layer_size if desired; here we assume it equals input_dim if not provided.
        'initial_layer_size': 128,
    }
    plugin_debug_vars = ['input_shape', 'encoding_dim']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var, None) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def configure_size(self, input_shape, interface_size, num_channels=None, use_sliding_windows=False):
        """
        Configures the transformer-based encoder.
        Args:
            input_shape (int): The length of the input sequence.
            interface_size (int): The dimension of the latent space.
            num_channels (int, optional): Number of input channels (default=1).
            use_sliding_windows (bool, optional): If True, input shape is (input_shape, num_channels); else, (input_shape, 1).
        """
        if num_channels is None:
            num_channels = 1
        self.params['input_shape'] = input_shape
        self.params['encoding_dim'] = interface_size
        self.params['num_channels'] = num_channels

        # Compute intermediate layer sizes as with the ANN plugin:
        intermediate_layers = self.params.get('intermediate_layers', 1)
        initial_layer_size = self.params.get('initial_layer_size', 128)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        ff_dim_divisor = self.params.get('ff_dim_divisor', 2)
        dropout_rate = self.params.get('dropout_rate', 0.1)
        learning_rate = self.params.get('learning_rate', 0.00001)

        layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, interface_size)
        layers.append(interface_size)
        print(f"[configure_size] Transformer Layer sizes: {layers}")
        print(f"[configure_size] Input sequence length: {input_shape}, Channels: {num_channels}")

        # Define input shape
        transformer_input_shape = (input_shape, num_channels)
        inputs = Input(shape=transformer_input_shape, name="encoder_input")
        x = inputs

        # For each intermediate layer, add a transformer block
        for size in layers[:-1]:
            ff_dim = max(size // ff_dim_divisor, 1)
            # Heuristic for number of attention heads
            if size < 64:
                num_heads = 2
            elif size < 128:
                num_heads = 4
            else:
                num_heads = 8
            # Transformer block:
            x = Dense(size, name="proj_dense")(x)
            x = MultiHeadAttention(head_num=num_heads, name="multi_head")(x)
            x = LayerNormalization(epsilon=1e-6, name="layer_norm_1")(x)
            x = Dropout(dropout_rate, name="dropout_1")(x)
            ffn_output = Dense(ff_dim, activation='relu', kernel_initializer=HeNormal(), name="ffn_dense_1")(x)
            ffn_output = Dense(size, name="ffn_dense_2")(ffn_output)
            ffn_output = Dropout(dropout_rate, name="dropout_2")(ffn_output)
            x = Add(name="residual_add")([x, ffn_output])
            x = LayerNormalization(epsilon=1e-6, name="layer_norm_2")(x)

        # Pool the sequence dimension and flatten
        x = GlobalAveragePooling1D()(x)
        x = Flatten()(x)
        outputs = Dense(interface_size, activation='tanh', kernel_initializer=GlorotUniform(), name="encoder_output")(x)

        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder_transformer")
        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Encoder Model Summary:")
        self.encoder_model.summary()

    def train(self, data):
        print(f"Training encoder with data shape: {data.shape}")
        self.encoder_model.fit(data, data, epochs=self.params.get('epochs',600),
                                batch_size=self.params.get('batch_size',128), verbose=1)
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

if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(input_shape=128, interface_size=4, num_channels=1, use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
