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
        Configures the CNN decoder.

        Args:
            interface_size (int): Latent space dimension for non-sliding mode.
                (Ignored in sliding-window mode.)
            input_shape (int or tuple): Original input shape fed to the encoder.
                - In non-sliding mode, it is an integer representing the number of features.
                - In sliding-window mode, it must be a tuple (window_size, num_features).
            num_channels (int): Number of input features/channels.
            encoder_output_shape: Output shape of the encoder (excluding batch size).
                - In sliding-window mode, this must be a tuple (sequence_length, latent_channels).
                - In non-sliding mode, it is ignored.
            use_sliding_windows (bool): Whether sliding windows are used.
        """
        self.params['input_shape'] = input_shape

        if not use_sliding_windows:
            # ===== Non-sliding mode (dense decoder) =====
            # Retrieve architecture parameters
            intermediate_layers = self.params.get('intermediate_layers', 3)
            initial_layer_size = self.params.get('initial_layer_size', 128)
            layer_size_divisor = self.params.get('layer_size_divisor', 2)
            l2_reg = self.params.get('l2_reg', 1e-5)
            learning_rate = self.params.get('learning_rate', 0.00002)
            
            # Compute encoder layer sizes as in the encoder
            encoder_layers = []
            current_size = initial_layer_size
            for i in range(intermediate_layers):
                encoder_layers.append(current_size)
                current_size = max(current_size // layer_size_divisor, 1)
            encoder_layers.append(interface_size)
            
            # Mirror the encoder intermediate layers
            decoder_intermediate_layers = encoder_layers[:-1][::-1]
            print(f"[configure_size] Encoder layer sizes: {encoder_layers}")
            print(f"[configure_size] Decoder intermediate layer sizes: {decoder_intermediate_layers}")

            # Determine final output units
            if isinstance(input_shape, tuple):
                final_output_units = int(np.prod(input_shape))
            else:
                final_output_units = input_shape

            # Build the decoder model using Dense layers
            from keras.layers import Input, Dense
            decoder_input = Input(shape=(interface_size,), name="decoder_input")
            x = decoder_input
            for idx, size in enumerate(decoder_intermediate_layers, start=1):
                x = Dense(units=size,
                        activation=LeakyReLU(alpha=0.1),
                        kernel_initializer=HeNormal(),
                        kernel_regularizer=l2(l2_reg),
                        name=f"decoder_dense_layer_{idx}")(x)
            x = Dense(units=final_output_units,
                    activation='linear',
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg),
                    name="decoder_output")(x)
            outputs = x
            self.model = Model(inputs=decoder_input, outputs=outputs, name="decoder_cnn")
            adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
            self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
            print("[configure_size] Decoder Model Summary (non-sliding branch):")
            self.model.summary()

        else:
            # ===== Sliding-window mode (conv transpose style) =====
            # In sliding-window mode, the decoder should take as input the encoder output shape.
            # Here we expect encoder_output_shape to be a tuple (sequence_length, latent_channels).
            if not (isinstance(encoder_output_shape, tuple) and len(encoder_output_shape) == 2):
                raise ValueError("[configure_size] In sliding windows mode, encoder_output_shape must be a tuple (sequence_length, latent_channels).")
            sequence_length, latent_channels = encoder_output_shape

            # The target output shape is given by input_shape, which must be a tuple (window_size, num_features)
            if not isinstance(input_shape, tuple):
                raise ValueError("[configure_size] In sliding windows mode, input_shape must be a tuple (window_size, num_features).")
            target_time_steps, target_channels = input_shape
            # For best results, the final output should have target_time_steps * target_channels units.
            final_output_units = target_time_steps * target_channels

            # Retrieve parameters
            learning_rate = self.params.get('learning_rate', 0.00002)
            l2_reg = self.params.get('l2_reg', 1e-5)

            # Build the decoder model:
            # The input to the decoder will be of shape encoder_output_shape.
            from keras.layers import Input, Dense, Reshape
            decoder_input = Input(shape=encoder_output_shape, name="decoder_input")
            x = decoder_input
            # Use a Dense layer to map the encoder output to the full reconstruction vector.
            x = Dense(units=final_output_units,
                    activation='linear',
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg),
                    name="decoder_dense_to_map")(x)
            # Reshape to (target_time_steps, target_channels)
            x = Reshape((target_time_steps, target_channels), name="decoder_final_reshape")(x)
            outputs = x

            self.model = Model(inputs=decoder_input, outputs=outputs, name="decoder_cnn")
            adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
            self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
            print("[configure_size] Decoder Model Summary (sliding-window branch):")
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
