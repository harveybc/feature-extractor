import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import keras.backend as K

class Plugin:
    """
    A decoder plugin using a simple neural network based on Keras, with dynamically configurable size.
    """

    # Define the parameters for this plugin and their default values
    plugin_params = {
        'encoding_dim': 32,
        'output_dim': None,
        'epochs': 50,
        'batch_size': 256
    }

    # Define the debug variables for this plugin
    plugin_debug_vars = ['encoding_dim', 'output_dim', 'epochs', 'batch_size']

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
            dict: Debug information including encoding_dim, output_dim, epochs, and batch_size.
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

    def configure_size(self, encoding_dim, output_dim):
        """
        Configure the decoder model architecture based on encoding and output dimensions.

        Args:
            encoding_dim (int): The size of the encoding layer (input to the decoder).
            output_dim (int): The size of the output layer (should match the original input dimension to the encoder).
        """
        self.params['encoding_dim'] = encoding_dim
        self.params['output_dim'] = output_dim

        input_layer = Input(shape=(encoding_dim,))
        decoded = Dense(int(encoding_dim * 2), activation='relu')(input_layer)
        decoded = Dense(output_dim, activation='sigmoid')(decoded)  # Final layer to reconstruct the original input

        self.model = Model(inputs=input_layer, outputs=decoded)
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, encoded_data, original_data):
        """
        Trains the decoder model on provided encoded data to reconstruct the original data.

        Args:
            encoded_data (np.array): Encoded data from the encoder.
            original_data (np.array): Original data to reconstruct.
        """
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=self.params['batch_size'])

    def decode(self, encoded_data):
        """
        Decodes the data using the trained model.

        Args:
            encoded_data (np.array): Encoded data to decode.

        Returns:
            np.array: Decoded (reconstructed) data.
        """
        return self.model.predict(encoded_data)

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
            reconstructed_data (np.array): The data after being decoded.

        Returns:
            float: The mean squared error.
        """
        return np.mean(np.square(original_data - reconstructed_data))
