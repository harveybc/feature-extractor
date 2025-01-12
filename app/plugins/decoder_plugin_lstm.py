import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


class Plugin:
    """
    A strictly 'inverse' decoder plugin that mirrors the encoder layer sizes in reverse.
    For example, if encoder = [LSTM(64), LSTM(32), Dense(16)],
    then decoder = [Dense(32), Dense(64), RepeatVector(36), LSTM(64), LSTM(32), TimeDistributed(1)].
    
    Adjust 'inverted_sizes' below to mirror your encoder exactly.
    """

    plugin_params = {
        # Training hyperparameters
        'epochs': 200,
        'batch_size': 128,

        # You can still keep some dropout / L2 if desired:
        'dropout_rate': 0.0,
        'l2_reg': 1e-4,

        # Learning rate
        'learning_rate': 0.0001,
    }

    # For debugging/tracking
    plugin_debug_vars = ['latent_dim', 'output_steps', 'output_channels']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def configure_size(
        self, 
        interface_size, 
        input_shape,           # <-- second arg matches the manager's call
        num_channels=None, 
        encoder_output_shape=None, 
        use_sliding_windows=False
    ):
        """
        Configures the decoder model with the specified latent dimension (interface_size) 
        and a second parameter (input_shape) which your AutoencoderManager is passing as 
        'input_shape' but in practice represents the 'number of time steps' to reconstruct.

        Args:
            interface_size (int): Size of the latent space (input dimension).
            input_shape (int): Number of time steps (or 'output_shape') the decoder should output.
            num_channels (int, optional): Number of features (channels) per time step.
            encoder_output_shape (tuple, optional): Shape of the encoder output (e.g., (latent_dim,)).
            use_sliding_windows (bool, optional): If True, input to the decoder is (interface_size, num_channels);
                                                otherwise it's just (interface_size,).
        """
        # Store the parameters internally
        self.params['interface_size'] = interface_size

        # Because the manager is passing 'input_shape' in the 2nd position, 
        # we interpret that as the time dimension to reconstruct
        self.params['output_shape'] = input_shape

        if num_channels is None:
            num_channels = 1  # default univariate

        # Decide the actual input shape to the decoder (the latent vector shape)
        if not use_sliding_windows:
            # shape is just (latent_dim,)
            decoder_input_shape = (interface_size,)
            print(f"[configure_size] Not using sliding windows: decoder_input_shape={decoder_input_shape}")
        else:
            # shape is (latent_dim, num_channels)
            decoder_input_shape = (interface_size, num_channels)
            print(f"[configure_size] Using sliding windows: decoder_input_shape={decoder_input_shape}")

        print(f"[configure_size] Decoder will reconstruct time steps={self.params['output_shape']}, "
            f"features={num_channels}")

        # Build the decoder model
        self.model = Sequential(name="decoder")

        # Example hyperparameters
        l2_reg = self.params.get('l2_reg', 1e-4)
        dropout_rate = self.params.get('dropout_rate', 0.0)
        learning_rate = self.params.get('learning_rate', 0.001)

        # 1) Dense layer to expand from latent_dim -> some initial hidden size
        initial_layer_size = self.params.get('initial_layer_size', 64)
        self.model.add(
            Dense(
                initial_layer_size,
                input_shape=decoder_input_shape,
                activation='relu',
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(l2_reg),
                name="decoder_input"
            )
        )

        # Optional dropout
        if dropout_rate > 0:
            self.model.add(Dropout(dropout_rate))

        # 2) RepeatVector to expand from (batch_size, hidden_dim) -> (batch_size, output_shape, hidden_dim)
        self.model.add(RepeatVector(self.params['output_shape']))
        print(f"[configure_size] Added RepeatVector layer with size: {self.params['output_shape']}")

        # 3) Build multiple LSTM layers that go from the initial_layer_size down 
        #    or however you'd like to define them.
        current_size = initial_layer_size
        intermediate_layers = self.params.get('intermediate_layers', 2)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)

        for i in range(intermediate_layers):
            # For "inverse" logic, you might keep or shrink the LSTM sizes as you like
            lstm_units = max(current_size, 8)  # ensure not too small
            print(f"[configure_size] Adding LSTM layer {i+1} with size={lstm_units}")

            self.model.add(
                LSTM(
                    units=lstm_units,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg),
                    return_sequences=True  # we want a full sequence for each time step
                )
            )

            if dropout_rate > 0.0:
                self.model.add(Dropout(dropout_rate))

            current_size = max(current_size // layer_size_divisor, 8)

        # 4) Final TimeDistributed Dense to produce (num_channels) features at each timestep
        self.model.add(
            TimeDistributed(
                Dense(
                    num_channels,
                    activation='tanh',
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg)
                ),
                name="decoder_output"
            )
        )
        print(f"[configure_size] Added TimeDistributed Dense layer with size={num_channels}")

        # Compile model
        adam_optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

        print("[configure_size] Decoder model compiled successfully.")
        self.model.summary()


    def train(self, encoded_data, original_data):
        """
        Train the inverse decoder with (encoded_data -> original_data).

        - encoded_data shape: (batch_size, latent_dim)
        - original_data shape: (batch_size, output_steps, output_channels)
        """
        if self.model is None:
            raise ValueError("[train] Decoder model is not configured. Call configure_size first.")

        print(f"[train] Training with encoded_data shape={encoded_data.shape}, original_data shape={original_data.shape}")
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']

        # If original_data is (batch_size, output_steps) for univariate, reshape => (batch_size, output_steps, 1)
        if len(original_data.shape) == 2:
            original_data = np.expand_dims(original_data, axis=-1)

        # Potentially add early stopping
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
        Decode latent vectors back into time-series sequences.
        
        Returns shape: (batch_size, output_steps, output_channels).
                       If output_channels=1, final shape => (batch_size, output_steps).
        """
        if self.model is None:
            raise ValueError("[decode] Decoder model is not configured.")

        print(f"[decode] Decoding data with shape={encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data)
        print(f"[decode] Decoded data shape={decoded_data.shape}")

        # If univariate => (batch_size, time_steps, 1), we can squeeze last dimension
        if decoded_data.shape[-1] == 1:
            decoded_data = decoded_data.squeeze(axis=-1)
        return decoded_data

    def calculate_mse(self, original_data, reconstructed_data):
        """
        Calculate MSE between original_data and reconstructed_data.
        Expects shapes => (batch_size, time_steps, 1) for univariate or (batch_size, time_steps, channels).
        """
        if len(original_data.shape) == 2:  # (batch_size, time_steps)
            original_data = np.expand_dims(original_data, axis=-1)
        if len(reconstructed_data.shape) == 2:  # (batch_size, time_steps)
            reconstructed_data = np.expand_dims(reconstructed_data, axis=-1)

        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"[calculate_mse] MSE={mse}")
        return mse

    def save(self, file_path):
        """
        Save the decoder model.
        """
        if self.model is None:
            raise ValueError("[save] Decoder model is not configured.")
        self.model.save(file_path)
        print(f"[save] Decoder model saved to {file_path}")

    def load(self, file_path):
        """
        Load a pre-trained decoder model from file.
        """
        self.model = load_model(file_path)
        print(f"[load] Decoder model loaded from {file_path}")


# Example usage
if __name__ == "__main__":
    # Suppose your encoder reduced shape (36,1) -> 16D latent
    # Then this decoder expands 16D latent -> shape (36,1).
    plugin = Plugin()
    plugin.configure_size(latent_dim=16, output_steps=36, output_channels=1)

    # Generate synthetic data
    batch_size = 8
    # Latent data: (batch_size, 16)
    encoded_data = np.random.randn(batch_size, 16)
    # Original data to reconstruct: (batch_size, 36, 1) => univariate
    original_data = np.random.randn(batch_size, 36)

    # Train
    plugin.train(encoded_data, original_data)

    # Decode
    decoded = plugin.decode(encoded_data)
    print(f"Decoded shape: {decoded.shape}")  
