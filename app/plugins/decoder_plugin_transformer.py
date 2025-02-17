import numpy as np
import tensorflow as tf
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Flatten, Reshape, GlobalAveragePooling1D, LayerNormalization, Dropout, Add, TimeDistributed, RepeatVector, Lambda
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber

# Reuse the positional encoding function from the encoder if needed.
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
    return pos_encoding

def add_positional_encoding(x):
    seq_len = tf.shape(x)[1]
    d_model = tf.shape(x)[2]
    pos_enc = positional_encoding(seq_len, d_model)
    return x + pos_enc

class Plugin:
    """
    A transformer-based decoder plugin.
    This decoder mirrors the encoder by first expanding the latent vector,
    repeating it to form a sequence, adding positional encoding,
    then applying transformer blocks in reverse order.
    """
    plugin_params = {
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'ff_dim_divisor': 2,
        'learning_rate': 0.00001,
        'dropout_rate': 0.1,
        'initial_layer_size': 128,
    }
    plugin_debug_vars = ['interface_size', 'output_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var, None) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def configure_size(self, interface_size, output_shape):
        """
        Configures the transformer-based decoder.
        Args:
            interface_size (int): Size of the latent vector (decoder input).
            output_shape (int): The desired dimension of the reconstructed (flattened) output.
        """
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        # Compute intermediate sizes similar to the encoder
        layer_sizes = []
        current_size = self.params.get('initial_layer_size', 128)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        int_layers = self.params.get('intermediate_layers', 1)
        for i in range(int_layers):
            layer_sizes.append(current_size)
            current_size = max(current_size // layer_size_divisor, interface_size)
        layer_sizes.append(interface_size)
        # Mirror the order for the decoder
        layer_sizes.reverse()
        print(f"[configure_size] Transformer decoder layer sizes (mirrored): {layer_sizes}")
        
        # Decoder input: latent vector (interface_size,)
        inputs = Input(shape=(interface_size,), name="decoder_input")
        # Expand to a sequence using RepeatVector to match the original sequence length.
        repeated = RepeatVector(output_shape, name="repeat_vector")(inputs)
        # Optionally add a Dense projection before transformer blocks.
        x = Dense(self.params.get('initial_layer_size', 128), activation='relu', name="proj_dense")(repeated)
        # Add fixed positional encoding so the model knows the order.
        x = Lambda(add_positional_encoding, name="positional_encoding")(x)
        
        dropout_rate = self.params.get('dropout_rate', 0.1)
        ff_dim_divisor = self.params.get('ff_dim_divisor', 2)
        # Apply transformer blocks for each layer size.
        for size in layer_sizes:
            ff_dim = max(size // ff_dim_divisor, 1)
            if size < 64:
                num_heads = 2
            elif size < 128:
                num_heads = 4
            else:
                num_heads = 8
            x = Dense(size, name="proj_dense_block")(x)
            x = MultiHeadAttention(head_num=num_heads, name="multi_head")(x)
            x = LayerNormalization(epsilon=1e-6, name="layer_norm_1")(x)
            x = Dropout(dropout_rate, name="dropout_1")(x)
            ffn_output = Dense(ff_dim, activation='relu', kernel_initializer=HeNormal(), name="ffn_dense_1")(x)
            ffn_output = Dense(size, name="ffn_dense_2")(ffn_output)
            ffn_output = Dropout(dropout_rate, name="dropout_2")(ffn_output)
            x = Add(name="residual_add")([x, ffn_output])
            x = LayerNormalization(epsilon=1e-6, name="layer_norm_2")(x)
        
        x = Flatten()(x)
        outputs = Dense(output_shape, activation='tanh', kernel_initializer=GlorotUniform(), name="decoder_output")(x)
        self.model = Model(inputs=inputs, outputs=outputs, name="decoder_transformer")
        adam_optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.00001), beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Transformer Decoder Model Summary:")
        self.model.summary()
