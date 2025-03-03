import numpy as np
from keras.models import Model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Input, Add, LayerNormalization, ZeroPadding1D
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class Plugin:
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-4,
        'activation': 'swish'
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def residual_block(self, x, skip, filters, name="res_block"):
        """
        Implements a residual convolutional block.
        Ensures the skip connection matches `x` in shape before merging.
        """
        print(f"[DEBUG] {name} - x shape: {x.shape}, skip shape: {skip.shape}")

        # Adjust depth if necessary
        if skip.shape[-1] != x.shape[-1]:
            print(f"[DEBUG] {name} - Projecting skip depth {skip.shape[-1]} -> {x.shape[-1]}")
            skip = Conv1D(filters=x.shape[-1], kernel_size=1, padding="same",
                          activation=None, kernel_initializer=HeNormal(),
                          kernel_regularizer=l2(self.params['l2_reg']),
                          name=f"{name}_skip_proj")(skip)

        # Convolution layers
        x = Conv1D(filters=filters, kernel_size=3, padding="same",
                   activation=self.params['activation'], kernel_initializer=HeNormal(),
                   kernel_regularizer=l2(self.params['l2_reg']),
                   name=f"{name}_conv1")(x)
        x = LayerNormalization(name=f"{name}_norm")(x)
        x = Conv1D(filters=filters, kernel_size=3, padding="same",
                   activation=self.params['activation'], kernel_initializer=HeNormal(),
                   kernel_regularizer=l2(self.params['l2_reg']),
                   name=f"{name}_conv2")(x)

        return Add(name=f"{name}_residual")([x, skip])  # Residual connection

    def build_decoder(self, latent_input, skip_tensors, output_shape, encoder_output_shape):
        """
        Builds the decoder model.
        """
        T, F = encoder_output_shape  
        window_size, orig_features = output_shape  

        x = Dense(units=T * F, activation=self.params['activation'],
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(self.params['l2_reg']),
                  name="latent_dense")(latent_input)
        x = Reshape((T, F), name="latent_reshape")(x)

        # Reverse the filters to match upsampling order
        filters = [self.params['initial_layer_size'] // (2 ** i) for i in range(self.params['intermediate_layers'])]
        filters.reverse()

        for idx, f in enumerate(filters):
            x = UpSampling1D(size=2, name=f"upsample_{idx+1}")(x)

            if skip_tensors and idx < len(skip_tensors):
                skip = skip_tensors[-(idx+1)]

                if skip.shape[1] != x.shape[1]:
                    pad_size = abs(skip.shape[1] - x.shape[1])
                    print(f"[DEBUG] Padding skip connection at {idx+1} with {pad_size} units")
                    skip = ZeroPadding1D(padding=(0, pad_size))(skip)

                x = self.residual_block(x, skip, filters=f, name=f"res_block_{idx+1}")

        x = Conv1D(filters=orig_features, kernel_size=1, activation='linear',
                   kernel_initializer=GlorotUniform(), kernel_regularizer=l2(self.params['l2_reg']),
                   name="decoder_final_conv")(x)

        return x

    def configure_size(self, interface_size, output_shape, num_channels, encoder_output_shape, use_sliding_windows, encoder_skip_connections):
        """
        Configures and builds the decoder model.
        """
        print(f"[DEBUG] Configuring decoder with interface_size={interface_size}, output_shape={output_shape}")

        latent_input = Input(shape=(interface_size,), name="decoder_latent")
        output = self.build_decoder(latent_input, encoder_skip_connections, output_shape, encoder_output_shape)

        self.model = Model(inputs=[latent_input] + encoder_skip_connections, outputs=output, name="decoder_model")
        print(f"[DEBUG] Final Output Shape: {self.model.output_shape}")

        self.model.compile(optimizer=Adam(learning_rate=self.params['learning_rate']),
                           loss=Huber(),
                           metrics=['mse', 'mae'])
        print("[DEBUG] Decoder model compiled successfully.")
