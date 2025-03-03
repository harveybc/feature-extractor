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
        'activation': 'swish'
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
        Ensures the skip connection matches `x` in shape before merging.
        """
        print(f"[DEBUG] residual_block - x shape: {x.shape}, skip shape: {skip.shape}")

        # Project skip connection if shape does not match
        if skip.shape[-1] != x.shape[-1]:
            print(f"[DEBUG] Adjusting skip connection from {skip.shape[-1]} to {x.shape[-1]} BEFORE merging")
            skip = Conv1D(filters=x.shape[-1], kernel_size=1, padding="same",
                          activation=None, kernel_initializer=HeNormal(),
                          kernel_regularizer=l2(self.params['l2_reg']),
                          name=f"{name}_skip_proj")(skip)
            skip = LayerNormalization(name=f"{name}_skip_norm")(skip)  # Normalize after projection

        print(f"[DEBUG] Pre-merge shapes: x {x.shape}, skip {skip.shape} at {name}")

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
        Builds the decoder model.
        """
        if isinstance(output_shape, tuple):
            window_size, orig_features = output_shape
        else:
            window_size = output_shape
            orig_features = None  

        T, F = encoder_output_shape  
        l2_reg = self.params.get('l2_reg', 1e-4)

        x = Dense(units=T * F, activation=self.params['activation'],
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(self.params['l2_reg']))(latent_input)
        x = Reshape((T, F), name="reshape")(x)

        enc_layers = []
        current = self.params['initial_layer_size']
        for i in range(self.params['intermediate_layers']):
            enc_layers.append(current)
            current = max(current // self.params['layer_size_divisor'], self.params['interface_size'])
        enc_layers.append(self.params['interface_size'])

        mirror_filters = enc_layers[:-1][::-1]  

        for idx in range(self.params['intermediate_layers']):
            x = UpSampling1D(size=2, name=f"upsample_{idx+1}")(x)

            if skip_tensors and idx < len(skip_tensors):
                skip = skip_tensors[-(idx+1)]

                if skip.shape[-1] != x.shape[-1]:
                    print(f"[DEBUG] Adjusting skip connection from {skip.shape[-1]} to {x.shape[-1]}")
                    skip = Conv1D(filters=x.shape[-1], kernel_size=1, padding="same",
                                  activation=None, kernel_initializer=HeNormal(),
                                  kernel_regularizer=l2(self.params['l2_reg']),
                                  name=f"skip_proj_{idx+1}")(skip)
                    skip = LayerNormalization(name=f"skip_norm_{idx+1}")(skip)  # Apply normalization after projection

                x = self.residual_block(x, skip, filters=mirror_filters[idx], dilation_rate=2, name=f"res_block_{idx+1}")
            else:
                x = Conv1D(filters=mirror_filters[idx], kernel_size=3, padding='same',
                           activation='tanh', kernel_initializer=HeNormal(),
                           kernel_regularizer=l2(self.params['l2_reg']),
                           name=f"conv1d_mirror_{idx+1}")(x)

        x = Conv1D(filters=orig_features, kernel_size=1, activation='linear',
                   kernel_initializer=GlorotUniform(),
                   kernel_regularizer=l2(self.params['l2_reg']),
                   name="decoder_final_conv")(x)

        return x

    def configure_size(self, interface_size, output_shape, num_channels, encoder_output_shape, use_sliding_windows, encoder_skip_connections):
        """
        Configures and builds the decoder model.
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
