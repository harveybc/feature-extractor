import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1DTranspose, GlobalAveragePooling1D, Dense, Input, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
# add reshape
from keras.layers import Reshape
import tensorflow as tf

class Plugin:
    """
    A decoder plugin using a convolutional neural network (CNN) based on Keras.
    This decoder mirrors the encoder's architecture: the intermediate filter sizes are 
    computed using the same parameters as in the encoder (initial_layer_size, intermediate_layers,
    and layer_size_divisor) but in reverse order. Batch normalization is omitted in the decoder
    for improved reconstruction flexibility.
    """
    plugin_params = {
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'l2_reg': 1e-5
    }
    plugin_debug_vars = ['input_shape', 'interface_size']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value 

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def configure_size(self, interface_size, input_shape, num_channels, encoder_output_shape, use_sliding_windows):
        """
        Configures the CNN decoder using Conv1DTranspose layers.
        
        Args:
            interface_size (int): Latent space dimension (decoder input size).
            input_shape (int or tuple): 
                - In non-sliding mode: an integer representing the original feature dimension.
                - In sliding-window mode: a tuple (window_size, num_features).
            num_channels (int): Number of input channels.
            encoder_output_shape: 
                - In sliding-window mode, a tuple (compressed_time, latent_channels) from the encoder.
                - In non-sliding mode, this is ignored.
            use_sliding_windows (bool): Whether sliding windows are used.
        """
        self.params['interface_size'] = interface_size
        self.params['input_shape'] = input_shape

        # Retrieve architecture parameters.
        intermediate_layers = self.params.get('intermediate_layers', 3)
        initial_layer_size = self.params.get('initial_layer_size', 128)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        l2_reg = self.params.get('l2_reg', 1e-5)
        learning_rate = self.params.get('learning_rate', 0.00002)

        # Determine target shape for reconstruction.
        if use_sliding_windows:
            if not isinstance(input_shape, tuple):
                raise ValueError("[configure_size] In sliding windows mode, input_shape must be a tuple (window_size, num_features).")
            target_time_steps, target_channels = input_shape
        else:
            # In non-sliding mode, treat input_shape as an integer; target shape becomes (input_shape, num_channels)
            if not isinstance(input_shape, int):
                raise ValueError("[configure_size] In non-sliding mode, input_shape must be an integer.")
            target_time_steps = input_shape
            target_channels = num_channels
        final_output_units = target_time_steps * target_channels
        print(f"[configure_size] (Non-sliding: {not use_sliding_windows}) Target shape: ({target_time_steps}, {target_channels})")

        # Compute encoder layer sizes as in the encoder.
        encoder_layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            encoder_layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        encoder_layers.append(interface_size)
        decoder_intermediate_layers = encoder_layers[:-1][::-1]
        print(f"[configure_size] Encoder layer sizes: {encoder_layers}")
        print(f"[configure_size] Decoder intermediate layer sizes: {decoder_intermediate_layers}")
        print(f"[configure_size] Original input shape: {input_shape}")

        # Determine the starting point for the decoder (i.e. the shape of the latent vector)
        if use_sliding_windows:
            # In sliding mode, we expect encoder_output_shape to be a tuple (compressed_time, latent_channels)
            if not (isinstance(encoder_output_shape, tuple) and len(encoder_output_shape) == 2):
                raise ValueError("[configure_size] In sliding windows mode, encoder_output_shape must be a tuple (compressed_time, latent_channels).")
            compressed_time, latent_channels = encoder_output_shape
        else:
            # For non-sliding mode, we set a default compressed time of 1 and latent_channels equal to interface_size.
            compressed_time = 1
            latent_channels = interface_size
        print(f"[configure_size] Using compressed_time={compressed_time}, latent_channels={latent_channels}")

        from keras.layers import Input, Conv1DTranspose, Reshape
        # For both modes, we set the decoder input shape as follows:
        if use_sliding_windows:
            decoder_input_shape = encoder_output_shape  # e.g. (compressed_time, latent_channels)
        else:
            decoder_input_shape = (interface_size,)
        decoder_input = Input(shape=decoder_input_shape, name="decoder_input")
        x = decoder_input

        # In the sliding mode, we upsample using Conv1DTranspose layers.
        # In non-sliding mode, we simulate a time dimension of 1.
        # We'll always use Conv1DTranspose layers here.
        # First, if in non-sliding mode, reshape the latent vector to (1, interface_size)
        if not use_sliding_windows:
            from keras.layers import Reshape
            x = Reshape((1, interface_size), name="decoder_initial_reshape")(x)
        
        # Now, apply a series of Conv1DTranspose layers to upsample the time dimension.
        # We target the final number of time steps = target_time_steps and channels = target_channels.
        # For simplicity, we use two upsampling layers (with strides=2) and then adjust with a final layer if needed.
        x = Conv1DTranspose(
                filters=decoder_intermediate_layers[0] if len(decoder_intermediate_layers) > 0 else latent_channels,
                kernel_size=3,
                strides=2,
                padding='same',
                activation=LeakyReLU(alpha=0.1),
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(l2_reg),
                name="decoder_conv1dtrans_1"
            )(x)
        x = Conv1DTranspose(
                filters=decoder_intermediate_layers[1] if len(decoder_intermediate_layers) > 1 else latent_channels,
                kernel_size=3,
                strides=2,
                padding='same',
                activation=LeakyReLU(alpha=0.1),
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(l2_reg),
                name="decoder_conv1dtrans_2"
            )(x)
        # (Optional) If a third upsampling is needed, apply it.
        current_time = int(x.shape[1])
        if current_time < target_time_steps:
            x = Conv1DTranspose(
                filters=decoder_intermediate_layers[2] if len(decoder_intermediate_layers) > 2 else latent_channels,
                kernel_size=3,
                strides=2,
                padding='same',
                activation=LeakyReLU(alpha=0.1),
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(l2_reg),
                name="decoder_conv1dtrans_3"
            )(x)
        # Final layer to map to the target number of channels.
        x = Conv1DTranspose(
                filters=target_channels,
                kernel_size=1,
                strides=1,
                padding='same',
                activation='linear',
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg),
                name="decoder_conv1dtrans_final"
            )(x)
        # At this point, x should have shape (None, new_time, target_channels).
        # We now force the time dimension to equal target_time_steps if necessary.
        current_time = int(x.shape[1])
        if current_time != target_time_steps:
            x = Reshape((target_time_steps, target_channels), name="decoder_final_reshape")(x)
            print(f"[configure_size] Reshaped decoder output to: ({target_time_steps}, {target_channels})")
        outputs = x

        self.model = Model(inputs=decoder_input, outputs=outputs, name="decoder_cnn")
        from keras.optimizers import Adam
        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Decoder Model Summary:")
        self.model.summary()



    def train(self, data, validation_data):
        if self.model is None:
            raise ValueError("[train] Decoder model is not configured. Call configure_size first.")
        print(f"[train] Starting training with data shape={data.shape}, validation shape={validation_data.shape}")
        history = self.model.fit(data, data, epochs=self.params.get('epochs',200),
                                  batch_size=self.params.get('batch_size',128),
                                  validation_data=(validation_data, validation_data),
                                  verbose=1)
        print("[train] Training completed.")
        return history

    def decode(self, encoded_data):
        if self.model is None:
            raise ValueError("[decode] Decoder model is not configured.")
        print(f"[decode] Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data, verbose=1)
        print(f"[decode] Decoded output shape: {decoded_data.shape}")
        return decoded_data

    def save(self, file_path):
        if self.model is None:
            raise ValueError("[save] Decoder model is not configured.")
        save_model(self.model, file_path)
        print(f"[save] Decoder model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"[load] Decoder model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    # For example, assume input_shape=128 (if sliding windows, it could be (window_size, num_channels))
    plugin.configure_size(interface_size=4, input_shape=128, num_channels=1, encoder_output_shape=(4,), use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
