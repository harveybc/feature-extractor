import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, BatchNormalization, Concatenate
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping
import tensorflow as tf

class Plugin:
    plugin_params = {
        'batch_size': 32,
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-4,
        'activation': 'tanh'
    }
    # For the encoder, "interface_size" is the latent dimension,
    # "output_shape" is the original input shape,
    # and "intermediate_layers" is preserved.
    plugin_debug_vars = ['interface_size', 'input_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None
        self.pre_flatten_shape = None  # Stores shape before flattening
        self.skip_connections = []     # Stores outputs before each pooling

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def configure_size(self, input_shape, interface_size, num_channels, use_sliding_windows):
        """
        Configures and builds the encoder model (as a Functional submodel) that mirrors the decoder.
        This model extracts a latent vector of dimension 'interface_size' from the input.
        
        Args:
            input_shape (tuple): Original input shape, e.g. (window_size, original_features).
            interface_size (int): The latent dimension.
            num_channels (int): Number of channels in the original input.
            use_sliding_windows (bool): Whether sliding windows are used.
        """
        if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
            raise ValueError(f"Invalid input_shape {input_shape}. Expected tuple (window_size, features).")
        self.params['input_shape'] = input_shape
        self.params['interface_size'] = interface_size
        print(f"[Encoder] Input shape: {input_shape}")

        # Determine the layer sizes (mirroring the decoder)
        # This creates a list: [initial_layer_size, ..., interface_size]
        layers = []
        current = self.params['initial_layer_size']
        for i in range(self.params['intermediate_layers']):
            layers.append(current)
            current = max(current // self.params['layer_size_divisor'], interface_size)
        layers.append(interface_size)
        print(f"[Encoder] Layer sizes: {layers}")

        inputs = Input(shape=input_shape, name="model_input")
        x = inputs

        # If using sliding windows, add positional encoding to the input.
        # This mirrors the decoder's addition of positional information.
        if self.params.get('use_pos_enc', False):
            def add_pos_enc(x):
                window_size = tf.shape(x)[1]
                positions = tf.range(start=0, limit=window_size, delta=1, dtype=tf.float32)
                positions = tf.expand_dims(positions, axis=1)  # (window_size, 1)
                feat_dim = tf.shape(x)[-1]
                i = tf.range(start=0, limit=feat_dim, delta=1, dtype=tf.float32)
                i = tf.expand_dims(i, axis=0)  # (1, feat_dim)
                angle_rates = 1 / tf.pow(10000.0, (2 * (tf.floor(i/2))) / tf.cast(feat_dim, tf.float32))
                angle_rads = tf.cast(positions, tf.float32) * angle_rates
                sinusoids = tf.concat([tf.sin(angle_rads[:, 0::2]), tf.cos(angle_rads[:, 1::2])], axis=-1)
                pos_encoding = tf.expand_dims(sinusoids, axis=0)  # (1, window_size, feat_dim)
                pos_encoding = tf.cast(pos_encoding, x.dtype)      # Cast to match x's dtype
                return x + pos_encoding

            x = tf.keras.layers.Lambda(add_pos_enc, name="encoder_positional_encoding")(x)
        # first  dense layer
        model_output = Dense(
            units=input_shape,
            activation='linear',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="linear_dense"
        )(x)
        x = BatchNormalization()(x)
        # Build convolutional blocks that downsample the input
        # Each block applies a Conv1D layer then downsampling via MaxPooling1D.
        self.skip_connections = []  # Reset skip connections (to be used by the decoder)
        l2_reg = self.params.get('l2_reg', 1e-4)
        for idx, size in enumerate(layers[:-1]):  # Exclude the final interface_size
            x = Conv1D(filters=size,
                       kernel_size=3,
                       activation=self.params['activation'],
                       kernel_initializer=HeNormal(),
                       padding='same',
                       kernel_regularizer=l2(l2_reg),
                       name=f"conv1d_{idx+1}")(x)
            # Store skip connection BEFORE pooling for later concatenation in the decoder.
            self.skip_connections.append(x)
            x = MaxPooling1D(pool_size=2, name=f"max_pool_{idx+1}")(x)

        # Apply batch normalization and a dense transformation to refine features.
        x = BatchNormalization(name="batch_norm1")(x)
        self.pre_flatten_shape = x.shape[1:]
        print(f"[Encoder] Pre-flatten shape: {self.pre_flatten_shape}")
        x = Flatten(name="flatten")(x)
        # Final Dense layer to produce the latent vector.
        model_output = Dense(units=layers[-1],
                             activation=self.params['activation'],
                             kernel_initializer=GlorotUniform(),
                             kernel_regularizer=l2(l2_reg),
                             name="model_output")(x)

        self.encoder_model = Model(inputs=inputs, outputs=model_output, name="encoder_cnn_model")
        print("[Encoder] CNN Model Summary:")
        self.encoder_model.summary()

        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )
        self.encoder_model.compile(optimizer=adam_optimizer,
                                   loss=Huber(),
                                   metrics=['mse', 'mae'],
                                   run_eagerly=False)
        print("[Encoder] Model compiled successfully.")

    def encode_data(self, data):
        """
        Encodes the data using the encoder model.
        
        Args:
            data (np.ndarray): Input data with appropriate shape.
            
        Returns:
            np.ndarray: Encoded latent representations.
        """
        print(f"[encode_data] Encoding data with shape: {data.shape}")
        try:
            encoded_data = self.encoder_model.predict(data)
            print(f"[encode_data] Encoded data shape: {encoded_data.shape}")
            return encoded_data
        except Exception as e:
            print(f"[encode_data] Exception occurred during encoding: {e}")
            raise ValueError("[encode_data] Failed to encode data. Please check model compatibility and data shape.")

    def save(self, file_path):
        """
        Saves the encoder model to the given file path.
        """
        save_model(self.encoder_model, file_path)
        print(f"[save] Encoder model saved to {file_path}")

    def load(self, file_path):
        """
        Loads the encoder model from the given file path.
        """
        self.encoder_model = load_model(file_path)
        print(f"[load] Encoder model loaded from {file_path}")
