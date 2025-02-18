import numpy as np
import tensorflow as tf
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Flatten, LayerNormalization, Add, Lambda
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal

# Force global policy to FP32.
tf.keras.mixed_precision.set_global_policy('float32')

def positional_encoding(seq_len, d_model):
    d_model_float = tf.cast(d_model, tf.float32)
    pos = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]  # shape (seq_len, 1)
    i = tf.cast(tf.range(d_model), tf.float32)[tf.newaxis, :]      # shape (1, d_model)
    angle_rates = 1 / tf.pow(10000.0, (2 * (tf.floor(i / 2)) / d_model_float))
    angle_rads = pos * angle_rates
    even_mask = tf.cast(tf.equal(tf.math.floormod(tf.range(d_model), 2), 0), tf.float32)
    even_mask = tf.reshape(even_mask, [1, d_model])
    pos_encoding = even_mask * tf.sin(angle_rads) + (1 - even_mask) * tf.cos(angle_rads)
    return pos_encoding

def add_positional_encoding(x):
    seq_len = tf.shape(x)[1]
    d_model = tf.shape(x)[2]
    pos_enc = positional_encoding(seq_len, d_model)
    pos_enc = tf.cast(pos_enc, tf.float32)
    return x + pos_enc

class Plugin:
    plugin_params = {
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'ff_dim_divisor': 2,
        'learning_rate': 0.00001,
        'dropout_rate': 0.0,  # No dropout for maximum accuracy
        'initial_layer_size': 128,
    }
    plugin_debug_vars = ['encoding_dim', 'input_shape', 'intermediate_layers']

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

    # Interface: configure_size(self, input_shape, encoding_dim, num_channels=None, use_sliding_windows=False)
    def configure_size(self, input_shape, encoding_dim, num_channels=None, use_sliding_windows=False):
        if num_channels is None:
            num_channels = 1
        self.params['input_shape'] = input_shape
        self.params['encoding_dim'] = encoding_dim
        self.params['num_channels'] = num_channels

        intermediate_layers = self.params.get('intermediate_layers', 1)
        initial_layer_size = self.params.get('initial_layer_size', 128)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        ff_dim_divisor = self.params.get('ff_dim_divisor', 2)
        learning_rate = self.params.get('learning_rate', 0.00001)

        # Compute transformer block sizes.
        layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, encoding_dim)
        layers.append(encoding_dim)
        print(f"[configure_size] Transformer Encoder Layer sizes: {layers}")
        print(f"[configure_size] Input sequence length: {input_shape}, Channels: {num_channels}")

        transformer_input_shape = (input_shape, num_channels)
        inputs = Input(shape=transformer_input_shape, name="encoder_input", dtype=tf.float32)
        x = inputs

        x = Lambda(add_positional_encoding, name="positional_encoding")(x)

        for size in layers[:-1]:
            ff_dim = max(size // ff_dim_divisor, 1)
            if size < 64:
                num_heads = 2
            elif size < 128:
                num_heads = 4
            else:
                num_heads = 8
            x = Dense(size)(x)
            x = MultiHeadAttention(head_num=num_heads, name=f"multi_head_{size}")(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            # Use tanh activation instead of ReLU in the feed-forward network.
            ffn_output = Dense(ff_dim, activation='tanh', kernel_initializer=HeNormal())(x)
            ffn_output = Dense(size)(ffn_output)
            x = Add()([x, ffn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

        x = Flatten()(x)
        outputs = Dense(encoding_dim, activation='linear', kernel_initializer=GlorotUniform(), name="encoder_output")(x)

        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder_transformer")
        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Encoder Model Summary:")
        self.encoder_model.summary()

if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(input_shape=128, encoding_dim=4, num_channels=1, use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
