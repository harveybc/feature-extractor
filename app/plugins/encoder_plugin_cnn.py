import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class Plugin:
    """
    A CNN-based encoder plugin for feature extraction using Keras.
    This model architecture is adapted from the perfect CNN_MMD predictor but modified
    so that its output dimension equals the desired interface size (i.e. the encoded representation).
    No MMD loss or metrics are included, since the autoencoder manager will add them if needed.
    """

    plugin_params = {
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-2,     # L2 regularization factor
        'activation': 'tanh'
    }

    plugin_debug_vars = ['interface_size', 'input_shape', 'intermediate_layers']

    def __init__(self):
        """
        Initializes the Plugin with default parameters and no model.
        """
        self.params = self.plugin_params.copy()
        self.encoder_model = None

    def set_params(self, **kwargs):
        """
        Updates the plugin parameters with provided keyword arguments.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Retrieves the current values of debug variables.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Adds the plugin's debug information to an external debug_info dictionary.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, input_shape, interface_size, num_channels, use_sliding_windows):
        """
        Configures and builds the encoder model.
        
        Parameters:
            input_shape (tuple): Must be of the form (window_size, features).
            interface_size (int): Desired dimension of the encoded representation.
            num_channels (int): Number of channels in the input data.
            use_sliding_windows (bool): Indicates whether sliding windows are used.
        
        The final model will accept input_shape and output a vector of size interface_size.
        """
        if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
            raise ValueError(f"Invalid input_shape {input_shape}. Expected tuple (window_size, features).")
        self.params['input_shape'] = input_shape
        self.params['interface_size'] = interface_size
        print(f"[configure_size] Encoder input_shape: {input_shape}, interface_size: {interface_size}, num_channels: {num_channels}, use_sliding_windows: {use_sliding_windows}")

        # Build layer size list using the perfect predictor approach.
        layers = []
        current_size = self.params['initial_layer_size']
        l2_reg = self.params.get('l2_reg', 1e-4)
        layer_size_divisor = self.params['layer_size_divisor']
        int_layers = 0
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
            int_layers += 1
        # For the encoder, the final output is the interface size.
        layers.append(interface_size)
        print(f"[configure_size] Encoder layer sizes: {layers}")

        # Define the Input layer
        inp = Input(shape=input_shape, name="encoder_input")
        x = inp
        # Initial Dense projection
        x = Dense(
            units=layers[0],
            activation=self.params['activation'],
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg)
        )(x)
        # Add intermediate Conv1D and MaxPooling1D layers with unique names
        for idx, size in enumerate(layers[:-1]):
            if size > 1:
                x = Conv1D(
                    filters=size, 
                    kernel_size=3, 
                    activation='relu', 
                    kernel_initializer=HeNormal(), 
                    padding='same',
                    kernel_regularizer=l2(l2_reg),
                    name=f"conv1d_{idx+1}"
                )(x)
                x = MaxPooling1D(pool_size=2, name=f"max_pool_{idx+1}")(x)
        # A final Dense layer before output
        x = Dense(
            units=layers[0],
            activation=self.params['activation'],
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="dense_final"
        )(x)
        x = BatchNormalization(name="batch_norm")(x)
        # Correctly apply the Flatten layer:
        flatten_layer = Flatten(name="flatten")
        x = flatten_layer(x)
        model_output = Dense(
            units=interface_size,
            activation='linear',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="encoder_output"
        )(x)

        # Create the encoder model
        self.encoder_model = Model(inputs=inp, outputs=model_output, name="cnn_mmd_encoder")
        print("Encoder Model Summary:")
        self.encoder_model.summary()

        # Define optimizer and compile the model with simple Huber loss
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )
        self.encoder_model.compile(
            optimizer=adam_optimizer,
            loss=Huber(),
            metrics=['mae'],
            run_eagerly=False
        )
        print("[configure_size] Encoder model compiled successfully.")



    def encode_data(self, data):
        """
        Encodes the input data using the encoder model.
        
        Parameters:
            data (numpy.ndarray): Input data.
        
        Returns:
            numpy.ndarray: Encoded representation.
        """
        print(f"[encode_data] Encoding data with shape: {data.shape}")
        try:
            encoded_data = self.encoder_model.predict(data)
            print(f"[encode_data] Encoded data shape: {encoded_data.shape}")
            return encoded_data
        except Exception as e:
            print(f"[encode_data] Exception occurred during encoding: {e}")
            raise ValueError("[encode_data] Failed to encode data. Please check model compatibility and data shape.")

    def save(self, file_path):
        """
        Saves the encoder model to the specified file path.
        """
        save_model(self.encoder_model, file_path)
        print(f"[save] Encoder model saved to {file_path}")

    def load(self, file_path):
        """
        Loads a trained encoder model from the specified file path.
        """
        self.encoder_model = load_model(file_path)
        print(f"[load] Encoder model loaded from {file_path}")
