import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Conv1D, UpSampling1D, BatchNormalization, Reshape, Concatenate, Flatten, Input
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping

class Plugin:
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3, 
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-2,
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
        and mirroring the encoder blocks.
        
        Args:
            latent_input: Keras tensor for the latent vector (shape: (None, interface_size)).
            skip_tensors: List of skip connection tensors from the encoder (ordered from first to last).
            output_shape (tuple): Original input shape, e.g. (window_size, original_features).
            encoder_output_shape (tuple): Encoder pre-flatten shape (T, F).
        
        Returns:
            Keras tensor for the decoder output.
        """
        # Extract window size and original feature count.
        if isinstance(output_shape, tuple):
            window_size, orig_features = output_shape
        else:
            window_size = output_shape
            orig_features = None  # Should not occur
        T, F = encoder_output_shape  # For example, (16, 32)
        flat_dim = T * F

        # Expand latent vector to flat_dim and reshape to (T, F)
        x = Dense(units=flat_dim,
                  activation=self.params['activation'],
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(self.params['l2_reg']))(latent_input)
        x = Reshape((T, F), name="reshape")(x)

        # Recompute encoder layer sizes as in the encoder.
        enc_layers = []
        current = self.params['initial_layer_size']
        for i in range(self.params['intermediate_layers']):
            enc_layers.append(current)
            current = max(current // self.params['layer_size_divisor'], self.params['interface_size'])
        enc_layers.append(self.params['interface_size'])
        # Mirror conv filter sizes from encoder conv blocks (exclude final Dense mapping)
        mirror_filters = enc_layers[:-1][::-1]  # E.g., if enc_layers = [128, 64, 32, 32] then mirror_filters = [32, 64, 128]
        
        # For each intermediate layer, upsample, concatenate corresponding skip, then apply Conv1D + BN with tanh.
        for idx in range(self.params['intermediate_layers']):
            x = UpSampling1D(size=2, name=f"upsample_{idx+1}")(x)
            if skip_tensors and idx < len(skip_tensors):
                skip = skip_tensors[-(idx+1)]
                x = Concatenate(axis=-1, name=f"skip_concat_{idx+1}")([x, skip])
            filt = mirror_filters[idx] if idx < len(mirror_filters) else mirror_filters[-1]
            x = Conv1D(filters=filt,
                       kernel_size=3,
                       padding='same',
                       activation='tanh',  # use tanh instead of LeakyReLU
                       kernel_initializer=HeNormal(),
                       kernel_regularizer=l2(self.params['l2_reg']),
                       name=f"conv1d_mirror_{idx+1}")(x)
            x = BatchNormalization(name=f"bn_decoder_{idx+1}")(x)
        # Final mapping: flatten and then use a Dense layer with linear activation.
        x = Flatten(name="decoder_flatten")(x)
        x = Dense(units=window_size * orig_features,
                  activation='linear',  # final mapping with linear activation
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(self.params['l2_reg']),
                  name="decoder_dense_output")(x)
        output = Reshape((window_size, orig_features), name="decoder_output")(x)
        return output

    def configure_size(self, interface_size, output_shape, num_channels, encoder_output_shape, use_sliding_windows, encoder_skip_connections):
        """
        Configures and builds the decoder model as the mirror of the encoder using skip connections.
        
        Args:
            interface_size (int): The latent dimension.
            output_shape (int or tuple): Original input shape; if tuple, first element is window size, second is feature count.
            num_channels (int): Number of channels in the original input.
            encoder_output_shape (tuple): Encoder pre-flatten shape (T, F).
            use_sliding_windows (bool): Whether sliding windows are used.
            encoder_skip_connections (list): List of skip connection tensors from the encoder.
        """
        # Ensure the latent dimension key is set.
        self.params['interface_size'] = interface_size

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
