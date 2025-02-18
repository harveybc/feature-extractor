import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LayerNormalization, Add, Lambda, Reshape
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal

# Set global precision to float32.
tf.keras.mixed_precision.set_global_policy('float32')

def positional_encoding(seq_len, d_model):
    d_model_float = tf.cast(d_model, tf.float32)
    pos = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
    i = tf.cast(tf.range(d_model), tf.float32)[tf.newaxis, :]
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
    # Interface: configure_size(self, input_shape, encoding_dim, num_channels=None, use_sliding_windows=False)
    plugin_params = {
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'ff_dim_divisor': 2,
        'learning_rate': 1e-5,
        'dropout_rate': 0.0,  # No dropout
        'initial_layer_size': 128,
    }
    plugin_debug_vars = ['encoding_dim', 'input_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def get_debug_info(self):
        return {k: self.params.get(k) for k in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def configure_size(self, input_shape, encoding_dim, num_channels=None, use_sliding_windows=False):
        # If not using sliding windows, assume one timestep.
        if num_channels is None:
            num_channels = 1
        time_steps = input_shape if use_sliding_windows else 1
        self.params['input_shape'] = time_steps
        self.params['encoding_dim'] = encoding_dim
        self.params['num_channels'] = num_channels

        inter_layers = self.params.get('intermediate_layers', 1)
        init_size = self.params.get('initial_layer_size', 128)
        layer_div = self.params.get('layer_size_divisor', 2)
        ff_div = self.params.get('ff_dim_divisor', 2)
        lr = self.params.get('learning_rate', 1e-5)

        # Compute transformer block sizes.
        sizes = []
        current = init_size
        for _ in range(inter_layers):
            sizes.append(current)
            current = max(current // layer_div, encoding_dim)
        sizes.append(encoding_dim)
        print(f"[configure_size] Transformer Encoder Layer sizes: {sizes}")
        print(f"[configure_size] Input sequence length: {time_steps}, Channels: {num_channels}")

        # Build model.
        inp = Input(shape=(time_steps, num_channels), dtype=tf.float32)
        x = Lambda(add_positional_encoding)(inp)
        for size in sizes[:-1]:
            ff_dim = max(size // ff_div, 1)
            num_heads = 2 if size < 64 else (4 if size < 128 else 8)
            x = Dense(size)(x)
            x = MultiHeadAttention(head_num=num_heads)(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            ffn = Dense(ff_dim, activation='tanh', kernel_initializer=HeNormal())(x)
            ffn = Dense(size)(ffn)
            x = Add()([x, ffn])
            x = LayerNormalization(epsilon=1e-6)(x)
        x = Flatten()(x)
        out = Dense(encoding_dim, activation='linear', kernel_initializer=GlorotUniform())(x)

        self.encoder_model = Model(inputs=inp, outputs=out)
        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.encoder_model.compile(optimizer=optimizer, loss='mean_squared_error')
        print("[configure_size] Encoder Model Summary:")
        self.encoder_model.summary()

if __name__ == "__main__":
    # For example, sliding windows with 256 timesteps and 8 channels.
    plugin = Plugin()
    plugin.configure_size(input_shape=256, encoding_dim=16, num_channels=8, use_sliding_windows=True)
    print("Debug Info:", plugin.get_debug_info())
