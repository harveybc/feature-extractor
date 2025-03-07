import numpy as np
import tensorflow as tf
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Flatten, LayerNormalization, Add, Lambda
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.regularizers import l2

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
    """
    Transformer-based Encoder Plugin updated to follow the structure of the working CNN encoder plugin.
    """
    plugin_params = {
        'batch_size': 32,
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'ff_dim_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-5,
        'activation': 'tanh'
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
        """
        Configures and builds the transformer encoder model.
        Expects input_shape as a tuple (time_steps, features) and does not use sliding windows.
        
        Args:
            input_shape (tuple): e.g. (sequence_length, features).
            encoding_dim (int): Latent dimension.
            num_channels (int, optional): Number of channels (defaults to 1 if not provided).
            use_sliding_windows (bool): Ignored for transformer.
        """
        if num_channels is None:
            num_channels = 1
        # For transformer, input_shape is used directly.
        self.params['input_shape'] = input_shape
        self.params['encoding_dim'] = encoding_dim
        self.params['num_channels'] = num_channels

        inter_layers = self.params.get('intermediate_layers', 3)
        init_size = self.params.get('initial_layer_size', 128)
        layer_div = self.params.get('layer_size_divisor', 2)
        ff_div = self.params.get('ff_dim_divisor', 2)
        lr = self.params.get('learning_rate', 0.0001)
        l2_reg = self.params.get('l2_reg', 1e-5)

        # Compute transformer block sizes.
        sizes = []
        current = init_size
        for _ in range(inter_layers):
            sizes.append(current)
            current = max(current // layer_div, encoding_dim)
        sizes.append(encoding_dim)
        print(f"[configure_size] Transformer Encoder Layer sizes: {sizes}")
        print(f"[configure_size] Input sequence length: {input_shape[0]}, Features: {input_shape[1]}")

        # Build the model.
        inp = Input(shape=input_shape, dtype=tf.float32, name="model_input")
        x = Lambda(add_positional_encoding, name="positional_encoding")(inp)
        # Build transformer blocks.
        for size in sizes[:-1]:
            ff_dim = max(size // ff_div, 1)
            num_heads = 2 if size < 64 else (4 if size < 128 else 8)
            x_proj = Dense(size, kernel_initializer=HeNormal(), kernel_regularizer=l2(l2_reg))(x)
            x_att = MultiHeadAttention(head_num=num_heads)(x_proj)
            x_norm = LayerNormalization(epsilon=1e-6)(x_att)
            ffn = Dense(ff_dim, activation='tanh', kernel_initializer=HeNormal())(x_norm)
            ffn = Dense(size, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(l2_reg))(ffn)
            x = Add()([x_norm, ffn])
            x = LayerNormalization(epsilon=1e-6)(x)
        x = Flatten()(x)
        out = Dense(encoding_dim, activation='linear', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(l2_reg))(x)

        self.encoder_model = Model(inputs=inp, outputs=out, name="transformer_encoder_model")
        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.encoder_model.compile(optimizer=optimizer, loss='mean_squared_error')
        print("[configure_size] Encoder Model Summary:")
        self.encoder_model.summary()

    def encode_data(self, data):
        print(f"[encode_data] Encoding data with shape: {data.shape}")
        try:
            encoded_data = self.encoder_model.predict(data)
            print(f"[encode_data] Encoded data shape: {encoded_data.shape}")
            return encoded_data
        except Exception as e:
            print(f"[encode_data] Exception occurred during encoding: {e}")
            raise ValueError("[encode_data] Failed to encode data. Please check model compatibility and data shape.")

    def save(self, file_path):
        save_model(self.encoder_model, file_path)
        print(f"[save] Encoder model saved to {file_path}")

    def load(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"[load] Encoder model loaded from {file_path}")

if __name__ == "__main__":
    # Example: input sequence of 256 timesteps with 8 features (channels). Note: Transformer encoder does not use sliding windows.
    plugin = Plugin()
    plugin.configure_size(input_shape=(256, 8), encoding_dim=16, num_channels=8, use_sliding_windows=False)
    print("Debug Info:", plugin.get_debug_info())
