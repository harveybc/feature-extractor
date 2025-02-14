import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

class Plugin:
    """
    A decoder plugin that mirrors the encoder's architecture.
    
    The decoder uses the inverse of the encoder's layers:
      - It first expands the latent vector (interface_size) via a Dense layer 
        to the size of the encoder's last LSTM units.
      - It then repeats this vector for the required number of time steps.
      - Finally, it applies a series of LSTM layers (with sizes reversed from the encoder)
        followed by a TimeDistributed Dense layer to reconstruct the original features.
    
    The encoder's layer sizes are computed using:
        initial_layer_size, intermediate_layers, and layer_size_divisor.
    """
    
    # Default hyperparameters (updated to match the new encoder)
    plugin_params = {
        'epochs': 200,
        'batch_size': 128,
        'dropout_rate': 0.1,
        'l2_reg': 1e-2,           # Matching encoder's default L2 regularization
        'learning_rate': 0.0001,   # Matching encoder's learning rate
        
        # These parameters must mirror the encoder's configuration:
        'initial_layer_size': 32,
        'intermediate_layers': 3,
        'layer_size_divisor': 2,
    }
    
    # List of variables to include in debugging info (if needed)
    plugin_debug_vars = []

    def __init__(self):
        """
        Initializes the plugin with default parameters.
        """
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided keyword arguments.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Returns a dictionary with debug information.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Extends an existing debug dictionary with the plugin's debug info.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, interface_size, input_shape, num_channels=None,
                       encoder_output_shape=None, use_sliding_windows=False):
        """
        Configures the decoder model by building an architecture that mirrors the encoder.
        
        Args:
            interface_size (int): Size of the latent space (matches encoder's interface_size).
            input_shape (int): Number of time steps to reconstruct (output sequence length).
            num_channels (int, optional): Number of features per time step. Defaults to 1 if None.
            encoder_output_shape (tuple, optional): Not used here.
            use_sliding_windows (bool, optional): 
                If True, the decoder's input shape is (interface_size, num_channels);
                otherwise, it is (interface_size,).
        """
        # Store key parameters
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = input_shape  # Number of time steps to reconstruct
        
        # Default to univariate if num_channels is not provided
        if num_channels is None:
            num_channels = 1

        # Determine the shape of the decoder's input (latent vector)
        if not use_sliding_windows:
            decoder_input_shape = (interface_size,)
            print(f"[configure_size] Not using sliding windows: decoder_input_shape={decoder_input_shape}")
        else:
            decoder_input_shape = (interface_size, num_channels)
            print(f"[configure_size] Using sliding windows: decoder_input_shape={decoder_input_shape}")

        print(f"[configure_size] Decoder will reconstruct {self.params['output_shape']} time steps with {num_channels} feature(s).")

        # ------------------------------------------------------------
        # 1. Compute the encoder's layer sizes to mirror them.
        # ------------------------------------------------------------
        initial_layer_size = self.params.get('initial_layer_size', 32)
        intermediate_layers = self.params.get('intermediate_layers', 3)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        
        # Compute encoder layers as done in the encoder plugin:
        #   encoder_layers = [initial_layer_size, initial_layer_size//divisor, ..., interface_size]
        encoder_layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            encoder_layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        encoder_layers.append(interface_size)
        print(f"[configure_size] Encoder layer sizes computed: {encoder_layers}")
        
        # The last LSTM layer in the encoder (before the final Dense) used units = encoder_layers[-2]
        latent_dense_units = encoder_layers[-2]
        
        # ------------------------------------------------------------
        # 2. Define decoder LSTM sizes as the mirror (reverse) of the encoder LSTM sizes.
        #    (We mirror only the LSTM part, i.e., the encoder_layers excluding the final Dense output.)
        # ------------------------------------------------------------
        decoder_lstm_sizes = list(reversed(encoder_layers[:-1]))
        print(f"[configure_size] Decoder LSTM sizes (mirrored): {decoder_lstm_sizes}")

        # ------------------------------------------------------------
        # 3. Build the decoder model.
        # ------------------------------------------------------------
        self.model = Sequential(name="decoder")
        
        # 3.1 Dense layer to map latent vector -> latent_dense_units
        self.model.add(
            Dense(
                latent_dense_units,
                input_shape=decoder_input_shape,
                activation='relu',  # Using ReLU for expansion; adjust if necessary.
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(self.params.get('l2_reg', 1e-2)),
                name="decoder_dense_expand"
            )
        )
        
        # Optional dropout after Dense expansion
        dropout_rate = self.params.get('dropout_rate', 0.0)
        if dropout_rate > 0:
            self.model.add(Dropout(dropout_rate, name="decoder_dropout_after_dense"))
        
        # 3.2 RepeatVector to expand the Dense output to a sequence
        self.model.add(RepeatVector(self.params['output_shape']))
        print(f"[configure_size] Added RepeatVector layer with output length: {self.params['output_shape']}")
        
        # 3.3 Add LSTM layers (mirrored from the encoder)
        for idx, units in enumerate(decoder_lstm_sizes):
            self.model.add(
                LSTM(
                    units=units,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    return_sequences=True,
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(self.params.get('l2_reg', 1e-2)),
                    name=f"decoder_lstm_{idx+1}"
                )
            )
            if dropout_rate > 0:
                self.model.add(Dropout(dropout_rate, name=f"decoder_dropout_after_lstm_{idx+1}"))
        
        # 3.4 Final TimeDistributed Dense layer to output the reconstructed features per time step.
        self.model.add(
            TimeDistributed(
                Dense(
                    num_channels,
                    activation='linear',  # Linear activation mirrors the encoder's Dense projection.
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(self.params.get('l2_reg', 1e-2))
                ),
                name="decoder_output"
            )
        )
        print(f"[configure_size] Added TimeDistributed Dense layer to produce {num_channels} feature(s) per time step.")
        
        # ------------------------------------------------------------
        # 4. Compile the model.
        # ------------------------------------------------------------
        adam_optimizer = Adam(
            learning_rate=self.params.get('learning_rate', 0.0001),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Decoder model compiled successfully.")
        self.model.summary()

    def train(self, encoded_data, original_data):
        """
        Train the decoder model using (encoded_data -> original_data).
        
        Args:
            encoded_data (np.ndarray): Latent vectors with shape (batch_size, interface_size)
              (if not using sliding windows).
            original_data (np.ndarray): The target sequences with shape 
              (batch_size, output_steps, output_channels).
        """
        if self.model is None:
            raise ValueError("[train] Decoder model is not configured. Call configure_size first.")
        
        print(f"[train] Training with encoded_data shape={encoded_data.shape}, original_data shape={original_data.shape}")
        epochs = self.params.get('epochs', 200)
        batch_size = self.params.get('batch_size', 128)
        
        # If original_data is univariate with shape (batch_size, output_steps),
        # reshape it to (batch_size, output_steps, 1)
        if len(original_data.shape) == 2:
            original_data = np.expand_dims(original_data, axis=-1)
        
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
            encoded_data (np.ndarray): Input latent vectors.
        
        Returns:
            np.ndarray: Reconstructed sequences with shape 
              (batch_size, output_steps, output_channels). If output_channels == 1,
              the last dimension is squeezed.
        """
        if self.model is None:
            raise ValueError("[decode] Decoder model is not configured.")
        
        print(f"[decode] Decoding data with shape={encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data)
        print(f"[decode] Decoded data shape={decoded_data.shape}")
        
        # If univariate, squeeze the last dimension
        if decoded_data.shape[-1] == 1:
            decoded_data = decoded_data.squeeze(axis=-1)
        return decoded_data

    def calculate_mse(self, original_data, reconstructed_data):
        """
        Calculates the Mean Squared Error (MSE) between the original and reconstructed data.
        
        Args:
            original_data (np.ndarray): Original data (shape: (batch_size, time_steps, channels) 
                or (batch_size, time_steps) for univariate).
            reconstructed_data (np.ndarray): Reconstructed data.
        
        Returns:
            float: The computed MSE.
        """
        if len(original_data.shape) == 2:
            original_data = np.expand_dims(original_data, axis=-1)
        if len(reconstructed_data.shape) == 2:
            reconstructed_data = np.expand_dims(reconstructed_data, axis=-1)
        
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"[calculate_mse] MSE={mse}")
        return mse

    def save(self, file_path):
        """
        Saves the decoder model to the specified file path.
        
        Args:
            file_path (str): The path where the model will be saved.
        """
        if self.model is None:
            raise ValueError("[save] Decoder model is not configured.")
        self.model.save(file_path)
        print(f"[save] Decoder model saved to {file_path}")

    def load(self, file_path):
        """
        Loads a pre-trained decoder model from the specified file path.
        
        Args:
            file_path (str): The path from which to load the model.
        """
        self.model = load_model(file_path)
        print(f"[load] Decoder model loaded from {file_path}")
