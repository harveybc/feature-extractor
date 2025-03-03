import numpy as np
from keras.models import Model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Concatenate, Input, Add, LayerNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class Plugin:
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,  # Configurable by the user
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-4,
        'activation': 'swish'  # Swish is smoother than tanh
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

    def residual_block(self, x, skip, filters, kernel_size=3, dilation_rate=1, name="res_block"):
        """
        Implements a residual convolutional block with dilation.
        If skip connection does not match the main tensor shape, a projection layer is applied.

        Args:
            x: Main input tensor.
            skip: Skip connection tensor from encoder.
            filters: Number of filters.
            kernel_size: Kernel size.
            dilation_rate: Dilation for expanded receptive field.
            name: Name of the block.
        """
        print(f"[DEBUG] residual_block - x shape: {x.shape}, skip shape: {skip.shape}")

        # Ensure skip connection matches x in last dimension
        if x.shape[-1] != skip.shape[-1]:
            print(f"[DEBUG] Mismatch detected! Projecting skip from {skip.shape[-1]} to {x.shape[-1]}")
            skip = Conv1D(filters=x.shape[-1], kernel_size=1, padding="same",
                          activation=None, kernel_initializer=HeNormal(),
                          kernel_regularizer=l2(self.params['l2_reg']),
                          name=f"{name}_proj")(skip)

        x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same",
                   dilation_rate=dilation_rate, activation=self.params['activation'],
                   kernel_initializer=HeNormal(), kernel_regularizer=l2(self.params['l2_reg']),
                   name=f"{name}_conv1")(x)
        x = LayerNormalization(name=f"{name}_norm")(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same",
                   dilation_rate=dilation_rate, activation=self.params['activation'],
                   kernel_initializer=HeNormal(), kernel_regularizer=l2(self.params['l2_reg']),
                   name=f"{name}_conv2")(x)

        return Add(name=f"{name}_residual")([x, skip])  # Residual connection

    def build_decoder(self, latent_input, skip_tensors, output_shape, encoder_output_shape):
        """
        Builds the decoder model with enhanced intermediate layers.
        
        Args:
            latent_input: Keras tensor for the latent vector (shape: (None, interface_size)).
            skip_tensors: List of skip connection tensors from the encoder.
            output_shape: Original input shape, e.g. (window_size, original_features).
            encoder_output_shape: Encoder pre-flatten shape (T, F).
        
        Returns:
            Keras tensor for the decoder output.
        """
        if isinstance(output_shape, tuple):
            window_size, orig_features = output_shape
        else:
            window_size = output_shape
            orig_features = None  # Should not occur

        T, F = encoder_output_shape  # e.g., (16, 32)
        l2_reg = self.params.get('l2_reg', 1e-4)

        # Expand latent vector to match encoder output shape
        x = Dense(units=T * F, activation=self.params['activation'],
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(self.params['l2_reg']))(latent_input)
        x = Reshape((T, F), name="reshape")(x)

        # Configure layer sizes
        enc_layers = []
        current = self.params['initial_layer_size']
        for i in range(self.params['intermediate_layers']):
            enc_layers.append(current)
            current = max(current // self.params['layer_size_divisor'], self.params['interface_size'])
        enc_layers.append(self.params['interface_size'])

        # Mirror filters with residual blocks
        mirror_filters = [max(f // 2, self.params['interface_size']) for f in enc_layers[:-1][::-1]]

        # Upsampling and applying stronger intermediate layers
        for idx in range(self.params['intermediate_layers']):
            x = UpSampling1D(size=2, name=f"upsample_{idx+1}")(x)

            if skip_tensors and idx < len(skip_tensors):
                skip = skip_tensors[-(idx+1)]

                # Ensure skip connection is the correct shape before merging
                if skip.shape[-1] != x.shape[-1]:
                    print(f"[DEBUG] Adjusting skip connection from {skip.shape[-1]} to {x.shape[-1]}")
                    skip = Conv1D(filters=x.shape[-1], kernel_size=1, padding="same",
                                  activation=None, kernel_initializer=HeNormal(),
                                  kernel_regularizer=l2(self.params['l2_reg']),
                                  name=f"skip_proj_{idx+1}")(skip)

                x = self.residual_block(x, skip, filters=mirror_filters[idx], dilation_rate=2, name=f"res_block_{idx+1}")
            else:
                filt = mirror_filters[idx] if idx < len(mirror_filters) else mirror_filters[-1]
                x = Conv1D(filters=filt, kernel_size=3, padding='same',
                           activation='tanh', kernel_initializer=HeNormal(),
                           kernel_regularizer=l2(self.params['l2_reg']),
                           name=f"conv1d_mirror_{idx+1}")(x)

        # Final Conv1D layer to ensure correct shape
        x = Conv1D(filters=orig_features, kernel_size=1, activation='linear',
                   kernel_initializer=GlorotUniform(),
                   kernel_regularizer=l2(self.params['l2_reg']),
                   name="decoder_final_conv")(x)

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

        if isinstance(output_shape, tuple):
            window_size, orig_features = output_shape
        else:
            window_size = output_shape
            orig_features = num_channels
        self.params['output_shape'] = window_size

        T, F = encoder_output_shape
        print(f"[DEBUG] Using encoder pre-flatten shape: T={T}, F={F}")
        print(f"[DEBUG] Starting decoder configuration with interface_size={interface_size}, output_shape={output_shape}, num_channels={num_channels}, encoder_output_shape={encoder_output_shape}, use_sliding_windows={use_sliding_windows}")

        latent_input = Input(shape=(interface_size,), name="decoder_latent")
        output = self.build_decoder(latent_input, encoder_skip_connections, output_shape, encoder_output_shape)
        self.model = Model(inputs=[latent_input] + encoder_skip_connections, outputs=output, name="decoder_cnn_model")
