import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

class Plugin:
    """
    A Bidirectional Long Short-Term Memory (Bi-LSTM) network-based decoder using Keras, with dynamically configurable size.
    """

    plugin_params = {
        'intermediate_layers': 2,
        'layer_size_divisor': 2,
        'learning_rate': 0.001,
        'dropout_rate': 0.1,
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
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)



    def configure_size(self, interface_size, output_shape, num_channels=None, encoder_output_shape=None, use_sliding_windows=False):
        """
        Configures the decoder model with the specified latent space size and output shape.

        Args:
            interface_size (int): Size of the latent space (input dimension).
            output_shape (int): Number of time steps in the output sequence.
            num_channels (int, optional): Number of output channels/features.
            encoder_output_shape (tuple, optional): Shape of the encoder's output.
            use_sliding_windows (bool, optional): Whether sliding windows are being used (default: False).
        """
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        # If we are NOT using sliding windows, we assume the latent vector is simply (interface_size,)
        # If we are using sliding windows, we assume (interface_size, num_channels)
        if not use_sliding_windows:
            input_shape = (interface_size,)
        else:
            input_shape = (interface_size, num_channels if num_channels else 1)

        # Example approach to define layer sizes:
        # We "grow" from interface_size up to output_shape
        layer_sizes = []
        current_size = interface_size
        while current_size < output_shape:
            layer_sizes.append(current_size)
            current_size = min(current_size * self.params['layer_size_divisor'], output_shape)
        layer_sizes.append(output_shape)

        print(f"[configure_size] Decoder layer sizes will be: {layer_sizes}")

        self.model = Sequential(name="decoder")

        # --- Dense layer to map from latent dim to some hidden dimension ---
        self.model.add(
            Dense(layer_sizes[0], input_shape=input_shape, activation='relu',
                kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01),
                name="decoder_input")
        )
        print(f"Added Dense layer with size {layer_sizes[0]} as decoder_input")

        # --- RepeatVector to match desired output timesteps ---
        self.model.add(RepeatVector(output_shape))
        print(f"Added RepeatVector layer with size: {output_shape}")

        # --- Add LSTM layers (all but last) ---
        for size in layer_sizes[:-1]:
            self.model.add(LSTM(units=size,
                                activation='tanh',
                                kernel_initializer=GlorotUniform(),
                                kernel_regularizer=l2(0.01),
                                return_sequences=True))
            print(f"Added LSTM layer with size {size}")

        # --- Final LSTM layer (return_sequences=True to output a full sequence) ---
        final_lstm_units = layer_sizes[-1]
        self.model.add(LSTM(units=final_lstm_units,
                            activation='tanh',
                            kernel_initializer=GlorotUniform(),
                            kernel_regularizer=l2(0.01),
                            return_sequences=True))
        print(f"Added final LSTM layer with size {final_lstm_units}")

        # --- Final TimeDistributed Dense to get the desired number of channels ---
        final_output_channels = num_channels if num_channels else 1
        self.model.add(TimeDistributed(
            Dense(final_output_channels, activation='tanh', kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(0.01))
        ))
        print(f"Added TimeDistributed Dense layer with size {final_output_channels}")

        # --- Compile the model ---
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Decoder model compiled successfully.")

    def train(self, encoded_data, original_data):
        """
        Trains the decoder on latent vectors (encoded_data) to reconstruct original_data.
        """
        if self.model is None:
            raise ValueError("[train] Decoder model is not configured.")

        print(f"[train] Training decoder with encoded data shape: {encoded_data.shape} "
            f"and original data shape: {original_data.shape}")

        # Flatten encoded vectors if needed: (batch_size, latent_dim)
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))

        # Reshape original data for LSTM: (batch_size, time_steps, num_channels)
        if len(original_data.shape) == 2:  # univariate, no sliding windows => shape (batch_size, time_steps)
            original_data = original_data.reshape((original_data.shape[0], original_data.shape[1], 1))
        elif len(original_data.shape) == 3:
            # Already in the form (batch_size, time_steps, features)
            pass

        self.model.fit(encoded_data, original_data,
                    epochs=self.params['epochs'],
                    batch_size=self.params['batch_size'],
                    verbose=1)
        print("[train] Training completed.")

    def decode(self, encoded_data):
        """
        Decodes latent vectors back into sequences.
        """
        if self.model is None:
            raise ValueError("[decode] Decoder model is not configured.")

        print(f"[decode] Decoding data with shape: {encoded_data.shape}")
        # Flatten to (batch_size, latent_dim)
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))

        # Predict
        decoded_data = self.model.predict(encoded_data)
        print(f"[decode] Decoded data shape: {decoded_data.shape}")

        # If univariate and shaped (batch_size, time_steps, 1), we can squeeze the last dimension
        if decoded_data.shape[-1] == 1:
            decoded_data = decoded_data.squeeze(axis=-1)
        return decoded_data




    def save(self, file_path):
        self.model.save(file_path)
        print(f"Decoder model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Decoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], original_data.shape[1], 1))  # Reshape to (batch_size, time_steps, 1)
        reconstructed_data = reconstructed_data.reshape((reconstructed_data.shape[0], reconstructed_data.shape[1], 1))  # Reshape to (batch_size, time_steps, 1)
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"Calculated MSE: {mse}")
        return mse

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=4, output_shape=32)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
