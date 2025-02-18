import numpy as np
import tensorflow as tf
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Flatten, LayerNormalization, Add, Lambda
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal

# New TensorFlow-based positional encoding function.
def positional_encoding(seq_len, d_model):
    # Create position indices (seq_len, 1) using tf.range.
    pos = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
    # Create dimension indices (1, d_model)
    i = tf.cast(tf.range(d_model), tf.float32)[tf.newaxis, :]
    # Compute angle rates using tf.pow.
    angle_rates = 1 / tf.pow(10000.0, (2 * (tf.floor(i / 2)) / d_model))
    angle_rads = pos * angle_rates
    # Create a mask for even indices.
    even_mask = tf.cast(tf.equal(tf.math.floormod(tf.range(d_model), 2), 0), tf.float32)
    even_mask = tf.reshape(even_mask, [1, d_model])
    # Apply sin to even indices and cos to odd indices.
    pos_encoding = even_mask * tf.sin(angle_rads) + (1 - even_mask) * tf.cos(angle_rads)
    return pos_encoding

class Plugin:
    plugin_params = {
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'ff_dim_divisor': 2,
        'learning_rate': 0.00001,
        'dropout_rate': 0.0,  # Dropout removed for maximum accuracy
        'initial_layer_size': 128,
    }
    plugin_debug_vars = ['interface_size', 'output_shape', 'intermediate_layers']

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

        intermediate_layers = self.params.get('intermediate_layers', 1)
        initial_layer_size = self.params.get('initial_layer_size', 128)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        ff_dim_divisor = self.params.get('ff_dim_divisor', 2)
        dropout_rate = self.params.get('dropout_rate', 0.0)
        learning_rate = self.params.get('learning_rate', 0.00001)

        # Compute transformer block sizes.
        layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, interface_size)
        layers.append(interface_size)
        print(f"[configure_size] Transformer Encoder Layer sizes: {layers}")
        print(f"[configure_size] Input sequence length: {input_shape}, Channels: {num_channels}")

        # Define input shape as (input_shape, num_channels)
        transformer_input_shape = (input_shape, num_channels)
        inputs = Input(shape=transformer_input_shape, name="encoder_input")
        x = inputs

        # Add fixed positional encoding using the TensorFlow-based function.
        def add_positional_encoding(x):
            seq_len = tf.shape(x)[1]
            d_model = tf.shape(x)[2]
            pos_enc = positional_encoding(seq_len, d_model)
            return x + pos_enc

        x = Lambda(add_positional_encoding, name="positional_encoding")(x)

        # Apply transformer blocks (with dropout removed) for each intermediate layer.
        for size in layers[:-1]:
            ff_dim = max(size // ff_dim_divisor, 1)
            if size < 64:
                num_heads = 2
            elif size < 128:
                num_heads = 4
            else:
                num_heads = 8
            x = Dense(size, name="proj_dense")(x)
            x = MultiHeadAttention(head_num=num_heads, name="multi_head")(x)
            x = LayerNormalization(epsilon=1e-6, name="layer_norm_1")(x)
            ffn_output = Dense(ff_dim, activation='relu', kernel_initializer=HeNormal(), name="ffn_dense_1")(x)
            ffn_output = Dense(size, name="ffn_dense_2")(ffn_output)
            x = Add(name="residual_add")([x, ffn_output])
            x = LayerNormalization(epsilon=1e-6, name="layer_norm_2")(x)

        x = Flatten()(x)
        outputs = Dense(interface_size, activation='linear', kernel_initializer=GlorotUniform(), name="encoder_output")(x)

        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder_transformer")
        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Encoder Model Summary:")
        self.encoder_model.summary()

if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(input_shape=128, interface_size=4, num_channels=1, use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
