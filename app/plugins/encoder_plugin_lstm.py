from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import numpy as np


class Plugin:
    """
    An encoder plugin using an LSTM network based on Keras,
    dynamically configurable for both sliding windows and single-row inputs.
    Incorporates ideas from your predictor snippet: flexible layer sizing, dropout, etc.
    """

    plugin_params = {
        # Training params
        'epochs': 200,
        'batch_size': 128,
        # Architecture params
        'intermediate_layers': 2,    # Number of LSTM layers
        'initial_layer_size': 64,    # Base # of hidden units in the first LSTM layer
        'layer_size_divisor': 2,     # Divisor for subsequent LSTM layers
        'learning_rate': 0.001,      # Learning rate for Adam optimizer
        'dropout_rate': 0.0,         # Dropout rate for regularization
        'l2_reg': 1e-4,              # L2 regularization factor
    }

    plugin_debug_vars = ['input_shape', 'intermediate_layers', 'epochs', 'batch_size']

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
                print("                Will assume 1 feature => shape = (time_steps, 1).")
                input_shape = (input_shape, 1)

        time_steps, features = input_shape
        self.params['input_shape'] = input_shape  # store for debugging

        # --- Build the LSTM-based encoder ---
        print(f"[configure_size] Building encoder with input_shape={input_shape}, interface_size={interface_size}")
        l2_reg = self.params.get('l2_reg', 1e-4)
        dropout_rate = self.params.get('dropout_rate', 0.0)
        initial_layer_size = self.params.get('initial_layer_size', 64)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        intermediate_layers = self.params.get('intermediate_layers', 2)

        # Input
        encoder_input = Input(shape=input_shape, name="encoder_input")

        # Dynamically create LSTM layers
        x = encoder_input
        current_size = initial_layer_size

        for i in range(intermediate_layers):
            # Example logic: from initial_layer_size down to at least interface_size
            lstm_units = max(current_size, interface_size)
            print(f"[configure_size] Adding LSTM layer {i+1} with size={lstm_units}")

            x = LSTM(
                units=lstm_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(l2_reg),
                return_sequences=True if i < (intermediate_layers - 1) else False
            )(x)

            # Optional dropout
            if dropout_rate > 0.0:
                x = Dropout(dropout_rate)(x)

            # Optional batch normalization (comment out if you don’t want it):
            # x = BatchNormalization()(x)

            # Decrease hidden size for next layer
            current_size = max(current_size // layer_size_divisor, interface_size)

        # After the final LSTM (which returns last output), project to interface_size
        encoder_output = Dense(interface_size, activation='tanh',
                               kernel_initializer=GlorotUniform(),
                               kernel_regularizer=l2(l2_reg),
                               name="encoder_output")(x)

        self.encoder_model = Model(inputs=encoder_input, outputs=encoder_output, name="encoder")

        # Compile
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Encoder Model Summary:")
        self.encoder_model.summary()

    def train(self, data, validation_data):
        """
        Trains the encoder model on the provided data in an autoencoder-like fashion: input->output = data.
        For a pure encoder, you'd typically combine with a decoder. But if you're
        just training the encoder alone, you can use data->data as a self-reconstruction loss.

        Args:
            data (np.ndarray): Training data of shape (batch_size, time_steps, features).
            validation_data (np.ndarray): Validation data of the same shape.
        """
        if self.encoder_model is None:
            raise ValueError("[train] Encoder model is not yet configured. Call configure_size first.")

        print(f"[train] Starting training with data shape={data.shape}, validation shape={validation_data.shape}")
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']

        # Early stopping
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
        Runs forward pass through the encoder to map data -> latent vectors.

        Args:
            data (np.ndarray): Shape (batch_size, time_steps, features).

        Returns:
            encoded_data (np.ndarray): Encoded latent vectors of shape (batch_size, interface_size).
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
