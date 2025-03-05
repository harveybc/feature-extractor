import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Conv1D, UpSampling1D, BatchNormalization, Reshape, Concatenate, Flatten, Input
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping
import tensorflow as tf

class Plugin:
    plugin_params = {
        'batch_size': 128,
        'interface_size': 4,
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-4,
        'activation': 'tanh'
    }
    plugin_debug_vars = ['interface_size', 'output_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def build_decoder(self, latent_input, skip_tensors, output_shape, encoder_output_shape):
        """
        Builds the decoder model (as a Functional submodel) by expanding the latent vector
        and mirroring the encoder blocks with reduced dimensionality.
        
        Args:
            latent_input: Keras tensor for the latent vector (shape: (None, interface_size)).
            skip_tensors: List of skip connection tensors from the encoder (ordered from first to last).
            output_shape (tuple): Original input shape, e.g. (window_size, original_features).
            encoder_output_shape (tuple): Encoder pre-flatten shape (T, F).
        
        Returns:
            Keras tensor for the decoder output.
        """
        # Extract window size and original feature count
        if isinstance(output_shape, tuple):
            window_size, orig_features = output_shape
        else:
            window_size = output_shape
            orig_features = None  # Should not occur
        T, F = encoder_output_shape  # e.g., (16, 32)
        l2_reg = self.params.get('l2_reg', 1e-4)
        # Expand latent vector to match the encoder output shape
        x = Dense(units=T * F, activation=self.params['activation'],
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(self.params['l2_reg']))(latent_input)
        x = Reshape((T, F), name="reshape")(x)

        # If sliding windows are used, add positional encoding to the reshaped tensor.
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

            x = tf.keras.layers.Lambda(add_pos_enc, name="decoder_positional_encoding")(x)

        # Reduce filter sizes in the decoder
        enc_layers = []
        current = self.params['initial_layer_size']
        for i in range(self.params['intermediate_layers']):
            enc_layers.append(current)
            current = max(current // self.params['layer_size_divisor'], self.params['interface_size'])
        enc_layers.append(self.params['interface_size'])
        
        # Mirror conv filter sizes with lightweight architecture
        mirror_filters = [max(f // 2, self.params['interface_size']) for f in enc_layers[:-1][::-1]]

        # Upsampling and mirroring the encoder structure with lightweight layers
        for idx in range(self.params['intermediate_layers']):
            x = UpSampling1D(size=2, name=f"upsample_{idx+1}")(x)
            if skip_tensors and idx < len(skip_tensors):
                skip = skip_tensors[-(idx+1)]
                x = Concatenate(axis=-1, name=f"skip_concat_{idx+1}")([x, skip])
            filt = mirror_filters[idx] if idx < len(mirror_filters) else mirror_filters[-1]
            x = Conv1D(filters=filt,
                       kernel_size=3,
                       padding='same',
                       activation=self.params['activation'],
                       kernel_initializer=HeNormal(),
                       kernel_regularizer=l2(l2_reg),
                       name=f"conv1d_mirror_{idx+1}")(x)

        # Final Conv1D layer to ensure proper channel alignment
        x = Conv1D(filters=orig_features, kernel_size=1, activation='linear',
                   kernel_initializer=GlorotUniform(),
                   kernel_regularizer=l2(l2_reg),
                   name="decoder_final_conv")(x)
        x = Dense(units=orig_features,
                  activation='linear',
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(l2_reg))(x)
        return x

    def configure_size(self, interface_size, output_shape, num_channels, encoder_output_shape, use_sliding_windows, encoder_skip_connections):
        """
        Configures and builds the decoder model as the mirror of the encoder using optimized architecture.
        
        Args:
            interface_size (int): The latent dimension.
            output_shape (int or tuple): Original input shape; if tuple, first element is window size, second is feature count.
            num_channels (int): Number of channels in the original input.
            encoder_output_shape (tuple): Encoder pre-flatten shape (T, F).
            use_sliding_windows (bool): Whether sliding windows are used.
            encoder_skip_connections (list): List of skip connection tensors from the encoder.
        """
        self.params['interface_size'] = interface_size
        self.params['use_sliding_windows'] = use_sliding_windows

        if isinstance(output_shape, tuple):
            window_size, orig_features = output_shape
        else:
            window_size = output_shape
            orig_features = num_channels
        self.params['output_shape'] = window_size

        if isinstance(encoder_output_shape, tuple) and len(encoder_output_shape) == 1:
            encoder_output_shape = (1, encoder_output_shape[0])
        T, F = encoder_output_shape
        print(f"[DEBUG] Using encoder pre-flatten shape: T={T}, F={F}")
        print(f"[DEBUG] Starting decoder configuration with interface_size={interface_size}, output_shape={output_shape}, num_channels={num_channels}, encoder_output_shape={encoder_output_shape}, use_sliding_windows={use_sliding_windows}")

        latent_input = Input(shape=(interface_size,), name="decoder_latent")
        output = self.build_decoder(latent_input, encoder_skip_connections, output_shape, encoder_output_shape)
        self.model = Model(inputs=[latent_input] + encoder_skip_connections, outputs=output, name="decoder_cnn_model")
        print(f"[DEBUG] Final Output Shape: {self.model.output_shape}")

        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-2,
            amsgrad=False
        )
        self.model.compile(optimizer=adam_optimizer,
                           loss=Huber(),
                           metrics=['mse', 'mae'],
                           run_eagerly=False)
        print(f"[DEBUG] Model compiled successfully.")
