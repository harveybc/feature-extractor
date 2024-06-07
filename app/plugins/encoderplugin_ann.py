import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K

class Plugin:
    """
    An encoder plugin using a simple neural network based on Keras, with dynamically configurable size.
    """

    # Define the parameters for this plugin and their default values
    plugin_params = {
        'input_dim': None,
        'encoding_dim': 32,
        'epochs': 50,
        'batch_size': 256
    }

    # Define the debug variables for this plugin
    plugin_debug_vars = ['input_dim', 'encoding_dim', 'epochs', 'batch_size']

    def __init__(self):
        """
        Initialize the Plugin with default parameters.
        """
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        """
        Set the parameters for the plugin.

        Args:
            **kwargs: Arbitrary keyword arguments for plugin parameters.
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        """
        Get debug information for the plugin.

        Returns:
            dict: Debug information including input_dim, encoding_dim, epochs, and batch_size.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Add plugin-specific debug information to the existing debug info.

        Args:
            debug_info (dict): The existing debug information dictionary.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, input_dim, encoding_dim):
        """
        Configure the encoder model architecture based on input and encoding dimensions.

        Args:
            input_dim (int): The number of input features.
            encoding_dim (int): The size of the encoding layer.
        """
        self.params['input_dim'] = input_dim
        self.params['encoding_dim'] = encoding_dim

        self.model = Sequential([
            Dense(encoding_dim, input_shape=(input_dim,), activation='relu'),
            Dense(int(encoding_dim / 2), activation='relu'),  # Intermediate compression
            Dense(input_dim, activation='sigmoid')  # Output layer to reconstruct the input
        ])
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, data):
        """
        Trains the encoder model on provided data.

        Args:
            data (np.array): Training data.
        """
        self.model.fit(data, data, epochs=self.params['epochs'], batch_size=self.params['batch_size'])

    def encode(self, data):
        """
        Encodes the data using the trained model.

        Args:
            data (np.array): Data to encode.

        Returns:
            np.array: Encoded data.
        """
        encoder_layer = self.model.layers[0]
        encoder_function = K.function([self.model.input], [encoder_layer.output])
        return encoder_function([data])[0]

    def save(self, file_path):
        """
        Saves the model to a specified path.

        Args:
            file_path (str): Path to save the model.
        """
        self.model.save(file_path)

    def load(self, file_path):
        """
        Loads a model from a specified path.

        Args:
            file_path (str): Path where the model is stored.
        """
        self.model = load_model(file_path)

    def calculate_mse(self, original_data, reconstructed_data):
        """
        Calculates the mean squared error between the original data and the reconstructed data.

        Args:
            original_data (np.array): The original data.
            reconstructed_data (np.array): The data after being encoded and then decoded.

        Returns:
            float: The mean squared error.
        """
        return np.mean(np.square(original_data - reconstructed_data))