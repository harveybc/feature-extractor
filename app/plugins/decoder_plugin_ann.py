import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2

class Plugin:
    """
    A decoder plugin using a dense neural network based on Keras.
    This model is built as the exact mirror of the encoder model.
    """

    plugin_params = {
        # Training parameters (if training decoder standalone)
        # Architecture parameters (must mirror the encoder's design)
        'intermediate_layers': 4,       # Number of dense layers before the final projection
        'initial_layer_size': 128,       # Base number of hidden units (from encoder)
        'layer_size_divisor': 2,        # Divisor to compute subsequent layer sizes
        'l2_reg': 1e-5,                 # L2 regularization factor
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
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, interface_size, input_shape, num_channels, encoder_output_shape, use_sliding_windows):
        """
        Configures the decoder model.
        
        Args:
            interface_size (int): The latent space dimension (decoder input size).
            input_shape (int or tuple): Original input shape fed to the encoder.
            num_channels (int): Number of input features/channels.
            encoder_output_shape (tuple): Output shape of the encoder (excluding batch size).
            use_sliding_windows (bool): Whether sliding windows are used.
        """
        self.params['interface_size'] = interface_size
        self.params['input_shape'] = input_shape

        # Retrieve architecture parameters
        intermediate_layers = self.params.get('intermediate_layers', 3)
        initial_layer_size = self.params.get('initial_layer_size', 32)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        l2_reg = self.params.get('l2_reg', 1e-2)
        learning_rate = self.params.get('learning_rate', 0.0001)

        # Recompute the encoder's layer sizes using the same logic as in the encoder
        # For the encoder, the layers were built as:
        #    [initial_layer_size, initial_layer_size//div, ..., interface_size]
        encoder_layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            encoder_layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        encoder_layers.append(interface_size)  # Final encoder layer equals latent dimension

        # For the decoder, mirror the encoder by reversing the intermediate layers.
        # (Do not include the latent layer itself in the reversal.)
        decoder_intermediate_layers = encoder_layers[:-1][::-1]

        print(f"[configure_size] Encoder layer sizes: {encoder_layers}")
        print(f"[configure_size] Decoder intermediate layer sizes: {decoder_intermediate_layers}")
        print(f"[configure_size] Original input shape: {input_shape}")

        # Determine final output units.
        # If sliding windows are used and input_shape is a tuple, output units = product(input_shape).
        if use_sliding_windows:
            if isinstance(input_shape, tuple):
                final_output_units = int(np.prod(input_shape))
            else:
                final_output_units = input_shape
        else:
            final_output_units = input_shape  # For non-sliding windows, input_shape is an integer

        # Build the decoder model.
        # The decoder input is the latent vector of size (interface_size,)
        decoder_input = Input(shape=(interface_size,), name="decoder_input")
        x = decoder_input

        # Add intermediate dense layers in the mirrored order.
        layer_idx = 0
        for size in decoder_intermediate_layers:
            layer_idx += 1
            x = Dense(
                units=size,
                activation=LeakyReLU(alpha=0.1),
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(l2_reg),
                name=f"decoder_dense_layer_{layer_idx}"
            )(x)
        #x = BatchNormalization(name=f"decoder_batch_norm_{layer_idx}")(x)

        # Final projection to reconstruct the original input.
        x = Dense(
            units=final_output_units,
            activation='linear',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="decoder_output"
        )(x)
        outputs = x
        #outputs = BatchNormalization(name="decoder_last_batch_norm")(x)

        self.model = Model(inputs=decoder_input, outputs=outputs, name="decoder")
        adam_optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

        print("[configure_size] Decoder Model Summary:")
        self.model.summary()

    def train(self, data, validation_data):
        """
        Trains the decoder model (if standalone training is needed).
        Typically, the decoder is trained as part of the autoencoder.
        
        Args:
            data (np.ndarray): Training data (latent vectors) of shape (batch_size, interface_size).
            validation_data (np.ndarray): Validation data (same shape).
        """
        if self.model is None:
            raise ValueError("[train] Decoder model is not yet configured. Call configure_size first.")

        print(f"[train] Starting training with data shape={data.shape}, validation shape={validation_data.shape}")
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']

        history = self.model.fit(
            data, data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, validation_data),
            verbose=1
        )
        print("[train] Training completed.")
        return history

    def decode(self, encoded_data):
        """
        Runs a forward pass through the decoder to reconstruct the original signal.
        
        Args:
            encoded_data (np.ndarray): Latent vectors of shape (batch_size, interface_size).
        
        Returns:
            np.ndarray: Reconstructed data (flattened output if sliding windows are not used).
        """
        if self.model is None:
            raise ValueError("[decode] Decoder model is not configured.")
        print(f"[decode] Decoding data with shape: {encoded_data.shape}")

        decoded_data = self.model.predict(encoded_data, verbose=1)
        print(f"[decode] Decoded output shape: {decoded_data.shape}")
        return decoded_data

    def save(self, file_path):
        """
        Saves the decoder model to the specified file path.
        """
        if self.model is None:
            raise ValueError("[save] Decoder model is not configured.")
        save_model(self.model, file_path)
        print(f"[save] Decoder model saved to {file_path}")

    def load(self, file_path):
        """
        Loads a pre-trained decoder model from the specified file path.
        """
        self.model = load_model(file_path)
        print(f"[load] Decoder model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    # For testing purposes, assume:
    #   - interface_size (latent dim) = 4,
    #   - input_shape = 128 (or a tuple like (window_size, num_channels) if sliding windows are used),
    #   - num_channels = 1,
    #   - encoder_output_shape = (4,), and
    #   - use_sliding_windows = False.
    plugin = Plugin()
    plugin.configure_size(interface_size=4, input_shape=128, num_channels=1, encoder_output_shape=(4,), use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
