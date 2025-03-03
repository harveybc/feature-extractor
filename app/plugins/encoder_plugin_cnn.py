import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, BatchNormalization, Dropout, LayerNormalization, MultiHeadAttention
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# New: Custom PositionalEncoding layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model

    def call(self, inputs):
        # inputs: (batch_size, window_size, num_features)
        window_size = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=window_size, delta=1, dtype=tf.float32)
        positions = tf.expand_dims(positions, axis=1)  # (window_size, 1)
        i = tf.range(start=0, limit=self.d_model, delta=1, dtype=tf.float32)
        i = tf.expand_dims(i, axis=0)  # (1, d_model)
        angle_rates = 1 / tf.pow(10000.0, (2 * (tf.floor(i/2))) / tf.cast(self.d_model, tf.float32))
        angle_rads = tf.cast(positions, tf.float32) * angle_rates  # (window_size, d_model)
        sinusoids = tf.concat([tf.sin(angle_rads[:, 0::2]), tf.cos(angle_rads[:, 1::2])], axis=-1)
        pos_encoding = tf.expand_dims(sinusoids, axis=0)  # (1, window_size, d_model)
        return inputs + pos_encoding

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"d_model": self.d_model})
        return config

class Plugin:
    """
    A CNN-based encoder plugin for feature extraction using Keras.
    This architecture is adapted from a CNN predictor and outputs a latent vector
    of dimension equal to the desired interface size.
    """
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-4,
        'activation': 'tanh',
        'dropout_rate': 0.1,
        'positional_encoding_dim': 16  # New parameter for positional encoding dimension
    }
    plugin_debug_vars = ['epochs', 'batch_size', 'input_shape', 'intermediate_layers', 'initial_layer_size', 'time_horizon']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None
        self.pre_flatten_shape = None  # Stores shape before flattening
        self.skip_connections = []     # Stores outputs before each pooling

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def configure_size(self, input_shape, interface_size, num_channels, use_sliding_windows):
        if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
            raise ValueError(f"Invalid input_shape {input_shape}. Expected tuple (window_size, features).")
        self.params['input_shape'] = input_shape
        self.params['time_horizon'] = interface_size
        print(f"CNN input_shape: {input_shape}")

        # Determine layer sizes
        layers = []
        current_size = self.params['initial_layer_size']
        l2_reg = self.params.get('l2_reg', 1e-4)
        layer_size_divisor = self.params['layer_size_divisor']
        int_layers = 0
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
            int_layers += 1
        layers.append(interface_size)
        print(f"CNN Layer sizes: {layers}")

        inputs = Input(shape=input_shape, name="model_input")
        x = inputs
        # If using sliding windows, add positional encoding
        if use_sliding_windows:
            d_model = self.params.get('positional_encoding_dim', 16)
            original_features = input_shape[1]
            # Optionally project input to d_model and back if dimensions differ
            if original_features != d_model:
                x = Dense(units=d_model, activation=self.params['activation'],
                          kernel_initializer=GlorotUniform(),
                          kernel_regularizer=l2(l2_reg),
                          name="proj_to_d_model")(x)
            x = PositionalEncoding(d_model=d_model, name="positional_encoding")(x)
            if original_features != d_model:
                x = Dense(units=original_features, activation=self.params['activation'],
                          kernel_initializer=GlorotUniform(),
                          kernel_regularizer=l2(l2_reg),
                          name="proj_back")(x)

        # First Dense layer uses linear activation for stability
        x = Dense(units=layers[0],
                  activation='linear',
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(l2_reg),
                  name="dense_initial")(x)
        self.skip_connections = []  # Reset skip connections

        # Convolution and pooling blocks with skip connection storage
        for idx, size in enumerate(layers[:-1]):
            if size > 1:
                x = Conv1D(filters=size,
                           kernel_size=3,
                           activation='tanh',
                           kernel_initializer=HeNormal(),
                           padding='same',
                           kernel_regularizer=l2(l2_reg),
                           name=f"conv1d_{idx+1}")(x)
                self.skip_connections.append(x)
                x = MaxPooling1D(pool_size=2, name=f"max_pool_{idx+1}")(x)

        # Batch normalization before the attention block
        x = BatchNormalization(name="batch_norm1")(x)

        # Insert a multi-head self-attention block to capture long-range dependencies
        attn_output = MultiHeadAttention(num_heads=4,
                                         key_dim=x.shape[-1],
                                         dropout=0.1,
                                         name="encoder_attention")(x, x)
        x = LayerNormalization(name="layer_norm_attn")(x + attn_output)
        x = Dropout(rate=self.params.get('dropout_rate', 0.1), name="dropout_attn")(x)

        # Final dense transformation before flattening
        x = Dense(units=layers[-2],
                  activation=self.params['activation'],
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(l2_reg),
                  name="dense_final")(x)
        x = BatchNormalization(name="batch_norm2")(x)
        self.pre_flatten_shape = x.shape[1:]
        print(f"[DEBUG] Pre-flatten shape: {self.pre_flatten_shape}")
        x = Flatten(name="flatten")(x)
        model_output = Dense(units=layers[-1],
                             activation='linear',
                             kernel_initializer=GlorotUniform(),
                             kernel_regularizer=l2(l2_reg),
                             name="model_output")(x)

        self.encoder_model = Model(inputs=inputs, outputs=model_output, name="encoder_cnn_model")
        print("CNN Model Summary:")
        self.encoder_model.summary()

        adam_optimizer = Adam(learning_rate=self.params['learning_rate'],
                              beta_1=0.9,
                              beta_2=0.999,
                              epsilon=1e-7,
                              amsgrad=False)
        self.encoder_model.compile(optimizer=adam_optimizer,
                                   loss=Huber(),
                                   metrics=['mse', 'mae'],
                                   run_eagerly=False)
        print("CNN Model compiled successfully.")

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
