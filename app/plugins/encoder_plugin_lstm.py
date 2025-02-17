# Revised Encoder Plugin Code
import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Dense, Input, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

class Plugin:
    """
    An encoder plugin using an LSTM network based on Keras.
    This version replicates the layer sizing and model architecture from the predictor plugin.
    """

    # Updated default parameters to mirror predictor's design
    plugin_params = {
        # Training params
        # Architecture params
        'intermediate_layers': 3,        # Number of LSTM layers before the final projection
        'initial_layer_size': 32,        # Base number of hidden units in the first LSTM layer
        'layer_size_divisor': 2,         # Divisor to compute subsequent layer sizes
        'l2_reg': 1e-2,                  # L2 regularization factor (matching predictor)
    }

    plugin_debug_vars = []  # You can add any variables you need to debug

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided keyword arguments.
        """
        for key, value in kwargs.items():
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

    def configure_size(self, input_shape, interface_size, num_channels=None, use_sliding_windows=False):
        """
        Configures the encoder model with the specified input shape and latent space size.
        This method builds an LSTM-based encoder model following the architecture of the predictor plugin.

        Args:
            input_shape (tuple or int): Shape of the input data (time_steps, num_channels) or just time_steps (int).
            interface_size (int): Size of the latent space (output dimension).
            num_channels (int, optional): Number of input features (if input_shape is an int).
            use_sliding_windows (bool, optional): If True, input_shape is (time_steps, num_channels).
        """
        # --- Handle input shape logic ---
        if isinstance(input_shape, int):
            if use_sliding_windows:
                input_shape = (input_shape, num_channels if num_channels else 1)
            else:
                print("[configure_size] WARNING: Received an int for input_shape without sliding windows.")
                print("                Assuming 1 feature => shape = (time_steps, 1).")
                input_shape = (input_shape, 1)

        self.params['input_shape'] = input_shape  # store for debugging

        # --- Build the LSTM-based encoder using predictor-style logic ---
        # Extract parameters
        intermediate_layers = self.params.get('intermediate_layers', 3)
        initial_layer_size = self.params.get('initial_layer_size', 32)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        l2_reg = self.params.get('l2_reg', 1e-2)
        learning_rate = self.params.get('learning_rate', 0.0001)

        # Build the list of layer sizes
        # For each intermediate LSTM, compute the number of units and then finally append interface_size.
        layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)  # Lower bound is 1
        layers.append(interface_size)  # Final output layer matches the latent dimension

        print(f"[configure_size] LSTM Layer sizes: {layers}")
        print(f"[configure_size] LSTM input shape: {input_shape}")

        # Input layer
        encoder_input = Input(shape=input_shape, name="encoder_input")
        x = encoder_input

        # Add LSTM layers with return_sequences=True for all layers except final Dense projection
        idx = 0
        for size in layers[:-1]:
            idx += 1
            if size > 1:
                x = LSTM(
                    units=size,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    return_sequences=True,
                    name=f"lstm_layer_{idx}"
                )(x)

        # Add a final LSTM layer without return_sequences.
        # It uses the penultimate layer size (i.e. layers[-2]) as units.
        if len(layers) >= 2:
            x = LSTM(
                units=layers[-2],
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=False,
                name="lstm_layer_final"
            )(x)
        else:
            # In case no intermediate layers are specified, this block would be reached.
            pass

        # Batch normalization after the final LSTM layer
        x = BatchNormalization(name="batch_norm_final")(x)

        # Final Dense layer to project into the latent space with activation 'linear'
        encoder_output = Dense(
            units=layers[-1],
            activation='linear',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="encoder_output"
        )(x)

        # Create and compile the model
        self.encoder_model = Model(inputs=encoder_input, outputs=encoder_output, name="encoder")
        adam_optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Encoder Model Summary:")
        self.encoder_model.summary()

    def train(self, data, validation_data):
        """
        Trains the encoder model on the provided data in a self-reconstruction fashion.

        Args:
            data (np.ndarray): Training data of shape (batch_size, time_steps, features).
            validation_data (np.ndarray): Validation data of the same shape.
        """
        if self.encoder_model is None:
            raise ValueError("[train] Encoder model is not yet configured. Call configure_size first.")

        print(f"[train] Starting training with data shape={data.shape}, validation shape={validation_data.shape}")
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.encoder_model.fit(
            data, data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, validation_data),
            callbacks=[early_stopping],
            verbose=1
        )

        print("[train] Training completed.")
        return history

    def encode(self, data):
        """
        Runs a forward pass through the encoder to map data to latent vectors.

        Args:
            data (np.ndarray): Input data of shape (batch_size, time_steps, features).

        Returns:
            np.ndarray: Encoded latent vectors of shape (batch_size, interface_size).
        """
        if self.encoder_model is None:
            raise ValueError("[encode] Encoder model is not configured.")
        print(f"[encode] Encoding data with shape: {data.shape}")

        encoded_data = self.encoder_model.predict(data, verbose=1)
        print(f"[encode] Encoded output shape: {encoded_data.shape}")
        return encoded_data

    def save(self, file_path):
        """
        Saves the encoder model to the specified file path.
        """
        if self.encoder_model is None:
            raise ValueError("[save] Encoder model is not configured.")
        save_model(self.encoder_model, file_path)
        print(f"[save] Encoder model saved to {file_path}")

    def load(self, file_path):
        """
        Loads a pre-trained encoder model from the specified file path.
        """
        self.encoder_model = load_model(file_path)
        print(f"[load] Encoder model loaded from {file_path}")
