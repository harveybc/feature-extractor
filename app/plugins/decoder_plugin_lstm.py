import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


class Plugin:
    """
    A decoder plugin (inverse of the encoder) that reconstructs time-series data
    from a latent vector using a flexible LSTM architecture in Keras.
    """

    plugin_params = {
        # Training hyperparameters
        'epochs': 200,
        'batch_size': 128,

        # Network architecture hyperparameters
        'initial_layer_size': 64,   # The first hidden dimension after Dense
        'intermediate_layers': 2,   # Number of LSTM layers
        'layer_size_divisor': 2,    # Divisor for hidden-layer sizes
        'learning_rate': 0.001,
        'dropout_rate': 0.1,
        'l2_reg': 1e-4,             # L2 regularization factor
    }

    plugin_debug_vars = ['interface_size', 'output_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided keyword arguments.
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        """
        Returns a subset of parameters for debugging/tracking.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Extends an existing debug dictionary with the plugin's debug info.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, interface_size, output_shape, num_channels=None, encoder_output_shape=None, use_sliding_windows=False):
        """
        Configures the decoder model with the specified latent dimension and output time-series shape.

        Args:
            interface_size (int): Size of the latent space (input dimension).
            output_shape (int): Number of time steps to reconstruct in the output sequence.
            num_channels (int, optional): Number of output features (channels).
            encoder_output_shape (tuple, optional): Shape of the encoder's output. (Not strictly needed here.)
            use_sliding_windows (bool, optional): If True, input shape is (interface_size, num_channels). 
                                                  Otherwise, (interface_size,).
        """
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape
        if num_channels is None:
            num_channels = 1  # Default to univariate if unspecified

        # Decide the input shape: latent vectors only
        if use_sliding_windows:
            input_shape = (interface_size, num_channels)
            print(f"[configure_size] Using sliding windows: input_shape={input_shape}")
        else:
            input_shape = (interface_size,)
            print(f"[configure_size] Not using sliding windows: input_shape={input_shape}")

        # --- Build a sequential decoder model ---
        self.model = Sequential(name="decoder")

        # --- 1) Dense layer to expand latent dim -> hidden dimension ---
        initial_layer_size = self.params['initial_layer_size']
        l2_reg = self.params['l2_reg']
        dropout_rate = self.params['dropout_rate']

        print(f"[configure_size] Adding Dense layer with size={initial_layer_size}")
        self.model.add(
            Dense(
                units=initial_layer_size,
                input_shape=input_shape,
                activation='relu',
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(l2_reg),
                name="decoder_input"
            )
        )

        # Optional: batch normalization after the Dense
        # self.model.add(BatchNormalization())

        # Optional: dropout after the Dense
        if dropout_rate > 0.0:
            self.model.add(Dropout(dropout_rate))

        # --- 2) RepeatVector to set the time dimension for LSTM ---
        print(f"[configure_size] Adding RepeatVector of size={output_shape}")
        self.model.add(RepeatVector(output_shape))

        # --- 3) Multiple LSTM layers ---
        # We start from `initial_layer_size` and keep dividing until we reach the final layer.
        current_size = initial_layer_size
        for i in range(self.params['intermediate_layers']):
            # Example logic: ensure we don't go below e.g. 4 or 8 or something
            lstm_units = max(current_size, 8)
            print(f"[configure_size] Adding LSTM layer {i+1} with size={lstm_units}")

            # Return sequences = True except for the last LSTM if you want 
            # a final LSTM to return a full sequence (for each time step).
            # Since we want a full sequence to feed into TimeDistributed at the end,
            # we keep return_sequences=True on all intermediate LSTMs
            self.model.add(
                LSTM(
                    units=lstm_units,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg),
                    return_sequences=True
                )
            )

            # Optional: batch normalization
            # self.model.add(BatchNormalization())

            # Optional: dropout
            if dropout_rate > 0.0:
                self.model.add(Dropout(dropout_rate))

            # Decrease hidden size for next LSTM
            current_size = max(current_size // self.params['layer_size_divisor'], 8)

        # --- 4) Final TimeDistributed Dense to produce the (num_channels) features per timestep ---
        print(f"[configure_size] Adding TimeDistributed Dense with size={num_channels}")
        self.model.add(
            TimeDistributed(
                Dense(
                    num_channels,
                    activation='tanh',
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg)
                )
            )
        )

        # --- 5) Compile the model ---
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

        print("[configure_size] Decoder Model Summary:")
        self.model.summary()

    def train(self, encoded_data, original_data):
        """
        Trains the decoder on latent vectors (encoded_data) to reconstruct original_data.

        Args:
            encoded_data (np.ndarray): shape = (batch_size, latent_dim) or (batch_size, latent_dim, channels) 
                                       if sliding windows.
            original_data (np.ndarray): shape = (batch_size, time_steps, num_channels).
        """
        if self.model is None:
            raise ValueError("[train] Decoder model is not configured.")

        # Flatten encoded vectors if needed
        # e.g. if shape is (batch_size, latent_dim, channels), flatten to (batch_size, latent_dim*channels)
        if len(encoded_data.shape) > 2:
            batch_size, dim1, dim2 = encoded_data.shape
            encoded_data = encoded_data.reshape(batch_size, dim1 * dim2)

        print(f"[train] Training decoder with encoded_data shape={encoded_data.shape} "
              f"and original_data shape={original_data.shape}")

        # Ensure original_data has shape (batch_size, time_steps, num_channels)
        if len(original_data.shape) == 2:
            # shape = (batch_size, time_steps) => univariate
            original_data = original_data.reshape((original_data.shape[0], original_data.shape[1], 1))
        elif len(original_data.shape) == 1:
            # shape = (batch_size,) => single time step
            original_data = original_data.reshape((original_data.shape[0], 1, 1))

        epochs = self.params['epochs']
        batch_size = self.params['batch_size']

        # Optional EarlyStopping
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        history = self.model.fit(
            encoded_data, original_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stopping]
        )
        print("[train] Training completed.")
        return history

    def decode(self, encoded_data):
        """
        Decodes latent vectors back into time-series sequences.

        Args:
            encoded_data (np.ndarray): shape = (batch_size, latent_dim) or (batch_size, latent_dim, channels).

        Returns:
            decoded_data (np.ndarray): shape = (batch_size, time_steps, num_channels).
                                       If univariate, shape => (batch_size, time_steps).
        """
        if self.model is None:
            raise ValueError("[decode] Decoder model is not configured.")

        print(f"[decode] Decoding data with shape: {encoded_data.shape}")
        # Flatten to (batch_size, latent_dim) if needed
        if len(encoded_data.shape) > 2:
            batch_size, dim1, dim2 = encoded_data.shape
            encoded_data = encoded_data.reshape(batch_size, dim1 * dim2)

        decoded_data = self.model.predict(encoded_data)
        print(f"[decode] Decoded data shape: {decoded_data.shape}")

        # If univariate => (batch_size, time_steps, 1), so we can squeeze the last dim
        if decoded_data.shape[-1] == 1:
            decoded_data = decoded_data.squeeze(axis=-1)

        return decoded_data

    def calculate_mse(self, original_data, reconstructed_data):
        """
        Calculate MSE between original and reconstructed, ensuring they match shapes.
        """
        if len(original_data.shape) == 2:
            # (batch_size, time_steps)
            original_data = original_data.reshape((original_data.shape[0], original_data.shape[1], 1))
        if len(reconstructed_data.shape) == 2:
            # (batch_size, time_steps)
            reconstructed_data = reconstructed_data.reshape((reconstructed_data.shape[0],
                                                             reconstructed_data.shape[1], 1))

        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"[calculate_mse] Calculated MSE: {mse}")
        return mse

    def save(self, file_path):
        """
        Saves the decoder model to the specified file path.
        """
        if self.model is None:
            raise ValueError("[save] Decoder model is not configured.")
        self.model.save(file_path)
        print(f"[save] Decoder model saved to {file_path}")

    def load(self, file_path):
        """
        Loads a pre-trained decoder model from the specified file path.
        """
        self.model = load_model(file_path)
        print(f"[load] Decoder model loaded from {file_path}")


# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=8, output_shape=32, num_channels=1, use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")

    # Fake data
    batch_size = 10
    latent_dim = 8
    time_steps = 32

    encoded_data = np.random.randn(batch_size, latent_dim)
    original_data = np.random.randn(batch_size, time_steps)  # univariate => (batch_size, time_steps)

    plugin.train(encoded_data, original_data)
    decoded = plugin.decode(encoded_data)
    print(f"Decoded shape: {decoded.shape}")
