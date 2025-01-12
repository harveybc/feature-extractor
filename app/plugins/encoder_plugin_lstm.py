from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Bidirectional, Dense, Input, Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


class Plugin:
    """
    An encoder plugin using a Bidirectional Long Short-Term Memory (Bi-LSTM) network based on Keras, 
    dynamically configurable for both sliding windows and single-row inputs.
    """

    plugin_params = {
        'intermediate_layers': 1,  # Number of LSTM layers
        'layer_size_divisor': 2,  # Factor by which layer sizes decrease
        'learning_rate': 0.0001,  # Learning rate for Adam optimizer
        'dropout_rate': 0.1,  # Dropout rate for regularization
    }

    plugin_debug_vars = ['input_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}


    def configure_size(self, input_shape, interface_size, num_channels=None, use_sliding_windows=False):
        """
        Configures the encoder model with the specified input shape and latent space size.
        
        Args:
            input_shape (tuple): Shape of the input data, e.g., (time_steps, num_features).
            interface_size (int): Size of the latent space (output dimensions).
            num_channels (int, optional): Number of input channels (default: None).
            use_sliding_windows (bool, optional): Whether sliding windows are being used (default: False).
        """
        # Ensure input_shape is a tuple for compatibility
        if isinstance(input_shape, int):
            input_shape = (input_shape, 1) if not use_sliding_windows else (input_shape, num_channels)

        self.params['input_shape'] = input_shape

        # Define input layer
        inputs = Input(shape=input_shape)

        # LSTM layers
        x = inputs
        current_size = input_shape[0]  # Time steps
        layer_sizes = []

        for i in range(self.params['intermediate_layers']):
            next_size = max(current_size // self.params['layer_size_divisor'], interface_size)
            layer_sizes.append(next_size)
            x = Bidirectional(LSTM(
                units=next_size,
                activation='tanh',
                return_sequences=(i < self.params['intermediate_layers'] - 1),
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(0.01)  # Added L2 regularization
            ))(x)
            current_size = next_size
        # print the layer sizes
        print(f"Layer sizes: {layer_sizes}")

        # Final Dense layer to project into latent space
        outputs = Dense(interface_size, activation='tanh', kernel_initializer=GlorotUniform())(x)

        # Define and compile the model
        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder")
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

        print(f"[configure_size] Encoder model configured with input shape {input_shape} and output size {interface_size}")
        self.encoder_model.summary()

        def train(self, data, validation_data):
            """
            Trains the encoder model with early stopping.
            
            Args:
                data (np.ndarray): Training data.
                validation_data (np.ndarray): Validation data.
            """
            print(f"Training encoder with data shape: {data.shape}")

            # Add early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            self.encoder_model.fit(
                data, data,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_data=(validation_data, validation_data),
                callbacks=[early_stopping],
                verbose=1
            )
            print("Training completed.")


    def encode(self, data):
        """
        Encodes the given data using the pre-configured encoder model.

        Args:
            data (np.ndarray): Input data to encode.

        Returns:
            np.ndarray: Encoded data.
        """
        if self.encoder_model is None:
            raise ValueError("[encode] Encoder model is not configured.")
        print(f"Encoding data with shape: {data.shape}")
        encoded_data = self.encoder_model.predict(data, verbose=1)
        print(f"Encoded data shape: {encoded_data.shape}")
        return encoded_data

    def save(self, file_path):
        """
        Saves the encoder model to the specified file path.
        
        Args:
            file_path (str): Path to save the encoder model.
        """
        save_model(self.encoder_model, file_path)
        print(f"Encoder model saved to {file_path}")

    def load(self, file_path):
        """
        Loads a pre-trained encoder model from the specified file path.
        
        Args:
            file_path (str): Path to load the encoder model from.
        """
        self.encoder_model = load_model(file_path)
        print(f"Encoder model loaded from {file_path}")
