import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LayerNormalization, Add, TimeDistributed, RepeatVector, Lambda
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal

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
    plugin_params = {
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'ff_dim_divisor': 2,
        'learning_rate': 1e-5,
        'dropout_rate': 0.0,
        'initial_layer_size': 128,
    }
    plugin_debug_vars = ['interface_size', 'output_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def get_debug_info(self):
        return {k: self.params.get(k) for k in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    # Interface: configure_size(self, interface_size, output_time_steps, num_channels=None, encoder_output_shape=None, use_sliding_windows=False)
    def configure_size(self, interface_size, output_time_steps, num_channels=None, encoder_output_shape=None, use_sliding_windows=False):
        self.params['interface_size'] = interface_size
        # When using sliding windows, output_time_steps is the number of timesteps;
        # otherwise, assume 1 timestep.
        time_steps = output_time_steps if use_sliding_windows else 1
        self.params['output_shape'] = time_steps
        if num_channels is None:
            num_channels = 1
        self.params['num_channels'] = num_channels
        self.params['encoder_output_shape'] = encoder_output_shape
        self.params['use_sliding_windows'] = use_sliding_windows

        init_size = self.params.get('initial_layer_size', 128)
        layer_div = self.params.get('layer_size_divisor', 2)
        inter_layers = self.params.get('intermediate_layers', 1)
        ff_div = self.params.get('ff_dim_divisor', 2)
        lr = self.params.get('learning_rate', 1e-5)

        # Compute layer sizes (mirroring the encoder).
        sizes = []
        current = init_size
        for _ in range(inter_layers):
            sizes.append(current)
            current = max(current // layer_div, interface_size)
        sizes.append(interface_size)
        sizes.reverse()
        print(f"[configure_size] Transformer decoder layer sizes (mirrored): {sizes}")

        inputs = Input(shape=(interface_size,), dtype=tf.float32)
        repeated = RepeatVector(time_steps)(inputs)
        x = Dense(init_size, activation='tanh')(repeated)
        x = Lambda(add_positional_encoding)(x)

        for size in sizes:
            ff_dim = max(size // ff_div, 1)
            if size < 64:
                num_heads = 2
            elif size < 128:
                num_heads = 4
            else:
                num_heads = 8
            x = Dense(size)(x)
            x = MultiHeadAttention(head_num=num_heads)(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            residual = x
            ffn = Dense(ff_dim, activation='tanh', kernel_initializer=HeNormal())(x)
            ffn = Dense(size)(ffn)
            x = Add()([residual, ffn])
            x = LayerNormalization(epsilon=1e-6)(x)

        x = Flatten()(x)
        outputs = Dense(time_steps, activation='linear', kernel_initializer=GlorotUniform())(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        print("[configure_size] Transformer Decoder Model Summary:")
        self.model.summary()

if __name__ == "__main__":
    plugin = Plugin()
    # Example: interface size 4, sliding window length 256, 1 channel.
    plugin.configure_size(interface_size=4, output_time_steps=256, num_channels=8, encoder_output_shape=None, use_sliding_windows=True)
    print("Debug Info:", plugin.get_debug_info())
