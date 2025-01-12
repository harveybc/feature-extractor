from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Bidirectional, Dense, Input, Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


class Plugin:
    """
    An encoder plugin using a Bidirectional Long Short-Term Memory (Bi-LSTM) network based on Keras, 
    dynamically configurable for both sliding windows and single-row inputs.
    """

    plugin_params = {
        'intermediate_layers': 2,  # Number of LSTM layers
        'layer_size_divisor': 2,  # Factor by which layer sizes decrease
        'learning_rate': 0.001,  # Learning rate for Adam optimizer
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
            input_shape (tuple or int): Shape of the input data (time_steps, num_channels) or time_steps.
            interface_size (int): Size of the latent space (output dimensions).
            num_channels (int, optional): Number of input channels/features.
            use_sliding_windows (bool, optional): Whether sliding windows are being used (default: False).
        """
        # --- Handle input shape for both sliding windows and single-row inputs ---
        if isinstance(input_shape, int):
            if use_sliding_windows:
                # If using sliding windows, treat input_shape as `time_steps` plus `num_channels`
                input_shape = (input_shape, num_channels if num_channels else 1)
            else:
                # If not using sliding windows, we generally expect a tuple (time_steps, num_features).
                # If you truly have only 1 feature, you can allow it, but let's print a warning:
                print("[configure_size] WARNING: You passed an integer input_shape without sliding windows.")
                print("                This will assume a single feature for each time step.")
                input_shape = (input_shape, 1)

        # Save final shape in plugin_params
        self.params['input_shape'] = input_shape
        time_steps, num_features = input_shape

        # --- Define input layer ---
        inputs = Input(shape=input_shape)

        # Instead of tying layer sizes to time_steps, pick a base hidden size
        # or use your existing logic but be mindful it ties hidden dims to time_steps.
        # Example: Let’s define a base hidden size = max(2*interface_size, 64)
        # (just as an example to avoid too-small or too-large hidden layers)
        base_hidden_size = max(2 * interface_size, 64)

        x = inputs
        layer_sizes = []
        current_hidden_size = base_hidden_size

        for i in range(self.params['intermediate_layers']):
            next_size = max(current_hidden_size // self.params['layer_size_divisor'], interface_size)
            layer_sizes.append(next_size)
            x = LSTM(
                next_size,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(0.01),
                return_sequences=True,
            )(x)
            current_hidden_size = next_size

        # Debugging: Print the calculated layer sizes
        print(f"[configure_size] LSTM layer sizes: {layer_sizes}")

        # --- Final LSTM layer (no return_sequences) ---
        x = LSTM(
            interface_size,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer=HeNormal(),
        )(x)

        # --- Final Dense layer to project into latent space ---
        outputs = Dense(interface_size, activation='tanh', kernel_initializer=GlorotUniform())(x)

        # --- Define and compile the model ---
        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder")
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

        print(f"[configure_size] Encoder configured with input {input_shape} -> output {interface_size}")
        self.encoder_model.summary()

    def train(self, data, validation_data):
        """
        Trains the encoder on the provided data.
        
        Args:
            data (np.array): Training data shaped (batch_size, time_steps, num_features)
            validation_data (np.array): Validation data with the same shape
        """
        if self.encoder_model is None:
            raise ValueError("[train] Encoder model is not configured.")

        print(f"[train] Training encoder with data shape: {data.shape}")

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.encoder_model.fit(
            data, data,  # Autoencoder-style training
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=(validation_data, validation_data),
            callbacks=[early_stopping],
            verbose=1
        )

        print("[train] Training completed.")
        return history

    def encode(self, data):
        """
        Encodes the data into latent space using the trained encoder model.
        """
        if self.encoder_model is None:
            raise ValueError("[encode] Encoder model is not configured.")
        print(f"[encode] Encoding data with shape: {data.shape}")

        encoded_data = self.encoder_model.predict(data, verbose=1)
        print(f"[encode] Encoded data shape: {encoded_data.shape}")
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
