import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Dense, Input, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

class Plugin:
    """
    An LSTM-based encoder plugin.
    Configurable similarly to the ANN plugin.
    """

    plugin_params = {
        'intermediate_layers': 3,        # Number of LSTM layers before the final projection
        'initial_layer_size': 32,        # Base hidden units in first LSTM layer
        'layer_size_divisor': 2,
        'l2_reg': 1e-2
    }

    plugin_debug_vars = ['input_shape', 'encoding_dim']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var, None) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def configure_size(self, interface_size, input_shape, num_channels, encoder_output_shape, use_sliding_windows):
        """
        Configures the CNN decoder.
        
        Args:
            interface_size (int): Latent space dimension (decoder input size).
            input_shape (int or tuple): 
                - In non-sliding mode, this is an integer (number of features).
                - In sliding-window mode, this must be a tuple (window_size, num_features).
            num_channels (int): Number of input features/channels.
            encoder_output_shape: Output shape of the encoder (excluding batch size).
                - In sliding-window mode, ideally a tuple (compressed_time, latent_channels).
            use_sliding_windows (bool): Whether sliding windows are used.
        """
        self.params['interface_size'] = interface_size
        self.params['input_shape'] = input_shape

        # Retrieve architecture parameters
        intermediate_layers = self.params.get('intermediate_layers', 3)
        initial_layer_size = self.params.get('initial_layer_size', 128)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        l2_reg = self.params.get('l2_reg', 1e-5)
        learning_rate = self.params.get('learning_rate', 0.00002)

        # Compute the encoder's layer sizes (as in the encoder)
        encoder_layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            encoder_layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        encoder_layers.append(interface_size)
        print(f"[configure_size] Encoder layer sizes (from encoder): {encoder_layers}")

        # Mirror the encoder: reverse the intermediate layers (exclude the latent interface itself)
        decoder_intermediate_layers = encoder_layers[:-1][::-1]
        print(f"[configure_size] Decoder intermediate layer sizes: {decoder_intermediate_layers}")
        print(f"[configure_size] Original input shape: {input_shape}")

        # Determine final output units:
        if use_sliding_windows and isinstance(input_shape, tuple):
            final_output_units = int(np.prod(input_shape))
        else:
            final_output_units = input_shape

        # In sliding-window mode, expect encoder_output_shape to be a tuple (compressed_time, latent_channels)
        if use_sliding_windows:
            if not (isinstance(encoder_output_shape, tuple) and len(encoder_output_shape) == 2):
                raise ValueError(f"[configure_size] In sliding windows mode, encoder_output_shape must be a tuple (compressed_time, latent_channels). Got {encoder_output_shape}")
            compressed_time, latent_channels = encoder_output_shape
            print(f"[configure_size] Using sliding window mode with compressed_time={compressed_time}, latent_channels={latent_channels}")
        else:
            # For non-sliding mode, we won't use encoder_output_shape
            compressed_time, latent_channels = (1, interface_size)

        from keras.layers import Input, Dense, Reshape, Conv1DTranspose, BatchNormalization, LeakyReLU
        # For sliding windows mode, we build a decoder using Conv1DTranspose layers.
        if use_sliding_windows:
            # Our target output shape is the original sliding-window shape.
            target_time_steps, target_channels = input_shape
            # Compute the total number of elements
            final_output_units = target_time_steps * target_channels

            # Start with an input layer for the decoder.
            decoder_input = Input(shape=encoder_output_shape, name="decoder_input")
            x = decoder_input

            # Here you can design the upsampling architecture.
            # For example, assume two upsampling layers to reach target_time_steps:
            x = Conv1DTranspose(filters=decoder_intermediate_layers[0],
                                kernel_size=3,
                                strides=2,
                                padding='same',
                                activation=LeakyReLU(alpha=0.1),
                                kernel_initializer=HeNormal(),
                                kernel_regularizer=l2(l2_reg),
                                name="decoder_conv1dtrans_1")(x)
            x = BatchNormalization(name="decoder_bn_1")(x)
            # A second Conv1DTranspose layer
            if len(decoder_intermediate_layers) > 1:
                x = Conv1DTranspose(filters=decoder_intermediate_layers[1],
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    activation=LeakyReLU(alpha=0.1),
                                    kernel_initializer=HeNormal(),
                                    kernel_regularizer=l2(l2_reg),
                                    name="decoder_conv1dtrans_2")(x)
                x = BatchNormalization(name="decoder_bn_2")(x)
            # Final layer to map to target_channels
            x = Conv1DTranspose(filters=target_channels,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                activation='linear',
                                kernel_initializer=GlorotUniform(),
                                kernel_regularizer=l2(l2_reg),
                                name="decoder_conv1dtrans_final")(x)
            # Now, x’s time dimension might not equal target_time_steps.
            # Use a Reshape layer to force the desired shape.
            # The current number of elements per sample must equal target_time_steps * target_channels.
            current_shape = x.shape.as_list()  # [None, current_time, target_channels]
            current_time = current_shape[1]
            if current_time != target_time_steps:
                x = Reshape((target_time_steps, target_channels), name="decoder_final_reshape")(x)
                print(f"[configure_size] Reshaped decoder output to: ({target_time_steps}, {target_channels})")
            outputs = x
            self.model = Model(inputs=decoder_input, outputs=outputs, name="decoder_cnn")
        else:
            # Non-sliding mode: mirror the encoder using Dense layers.
            from keras.layers import Input, Dense
            decoder_input = Input(shape=(interface_size,), name="decoder_input")
            x = decoder_input
            # Compute mirrored intermediate layers
            # (Here we use the same dense mirror approach as in your ANN decoder)
            # Compute final output units = input_shape (an integer)
            if isinstance(input_shape, tuple):
                final_output_units = int(np.prod(input_shape))
            else:
                final_output_units = input_shape
            # Compute encoder layer sizes as before
            encoder_layers = []
            current_size = initial_layer_size
            for i in range(intermediate_layers):
                encoder_layers.append(current_size)
                current_size = max(current_size // layer_size_divisor, 1)
            encoder_layers.append(interface_size)
            decoder_intermediate_layers = encoder_layers[:-1][::-1]
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
        from keras.optimizers import Adam
        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Decoder Model Summary:")
        self.model.summary()



    def train(self, data, validation_data):
        if self.encoder_model is None:
            raise ValueError("[train] Encoder model is not yet configured. Call configure_size first.")
        print(f"[train] Starting training with data shape={data.shape}, validation shape={validation_data.shape}")
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.encoder_model.fit(data, data, epochs=self.params.get('epochs',600),
                                          batch_size=self.params.get('batch_size',128),
                                          validation_data=(validation_data, validation_data),
                                          callbacks=[early_stopping], verbose=1)
        print("[train] Training completed.")
        return history

    def encode(self, data):
        if self.encoder_model is None:
            raise ValueError("[encode] Encoder model is not configured.")
        print(f"[encode] Encoding data with shape: {data.shape}")
        encoded_data = self.encoder_model.predict(data, verbose=1)
        print(f"[encode] Encoded output shape: {encoded_data.shape}")
        return encoded_data

    def save(self, file_path):
        if self.encoder_model is None:
            raise ValueError("[save] Encoder model is not configured.")
        save_model(self.encoder_model, file_path)
        print(f"[save] Encoder model saved to {file_path}")

    def load(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"[load] Encoder model loaded from {file_path}")

if __name__ == "__main__":
    # For testing: assume 128 time steps and 8 features (non-sliding mode)
    plugin = Plugin()
    plugin.configure_size(input_shape=128, encoding_dim=4, num_channels=8, use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
