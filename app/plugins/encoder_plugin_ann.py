import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

class Plugin:
    """
    An encoder plugin using a dense neural network based on Keras, updated to use
    the same layer sizing system and architectural ideas as the LSTM encoder plugin.
    """

    plugin_params = {
        # Training parameter
        # Architecture parameters
        'intermediate_layers': 2,       # Number of dense layers before the final projection
        'initial_layer_size': 144,       # Base number of hidden units in the first dense layer
        'layer_size_divisor': 3,        # Divisor for subsequent layer sizes
        'l2_reg': 1e-5,                 # L2 regularization factor
    }

    plugin_debug_vars = ['input_dim', 'encoding_dim']

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

    def configure_size(self, input_dim, encoding_dim, num_channels=None, use_sliding_windows=False):
        """
        Configures the encoder model with the specified input dimension and latent space size.
        The intermediate layer sizes are calculated using the same system as in the LSTM plugin.

        Args:
            input_dim (int): Dimensionality of the input features.
            encoding_dim (int): Size of the latent space (output dimension).
        """
        self.params['input_dim'] = input_dim
        self.params['encoding_dim'] = encoding_dim

        # Extract parameters
        intermediate_layers = self.params.get('intermediate_layers', 3)
        initial_layer_size = self.params.get('initial_layer_size', 32)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        l2_reg = self.params.get('l2_reg', 1e-2)
        learning_rate = self.params.get('learning_rate', 0.0001)

        # Build the list of dense layer sizes following the LSTM plugin's logic.
        # Start with the initial_layer_size and divide successively, then append the encoding_dim.
        layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)  # Lower bound is 1
        layers.append(encoding_dim)  # Final output layer matches the latent dimension

        print(f"[configure_size] Dense Layer sizes: {layers}")
        print(f"[configure_size] Input dimension: {input_dim}")

        # Input layer
        inputs = Input(shape=(input_dim,), name="encoder_input")
        x = inputs

        # Add intermediate dense layers
        layer_idx = 0
        for size in layers[:-1]:
            layer_idx += 1
            x = Dense(
                units=size,
                activation='linear',
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(l2_reg),
                name=f"dense_layer_{layer_idx}"
            )(x)
        x = BatchNormalization(name=f"batch_norm_{layer_idx}")(x)
 
        # Final Dense layer to project into the latent space with linear activation
        x = Dense(
            units=layers[-1],
            activation="tanh",
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="encoder_output"
        )(x)
        outputs = x
        #outputs = BatchNormalization(name="encoder_last_batch_norm")(x)

        # Create and compile the model
        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder_dense")
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
            data (np.ndarray): Training data of shape (batch_size, input_dim).
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
            data (np.ndarray): Input data of shape (batch_size, input_dim).

        Returns:
            np.ndarray: Encoded latent vectors of shape (batch_size, encoding_dim).
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

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(input_dim=128, encoding_dim=4)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
