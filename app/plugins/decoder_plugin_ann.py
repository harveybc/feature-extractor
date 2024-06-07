import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

class DefaultDecoder:
    """
    A simple neural network based decoder using Keras, with dynamically configurable size.
    """
    def __init__(self):
        """
        Initializes the DefaultDecoder without a fixed architecture, to be configured later.
        """
        self.model = None

    def configure_size(self, encoding_dim, output_dim):
        """
        Configure the decoder model architecture based on encoding and output dimensions.

        Args:
            encoding_dim (int): The size of the encoding layer (input to the decoder).
            output_dim (int): The size of the output layer (should match the original input dimension to the encoder).
        """
        # Initialize the model architecture
        self.model = Sequential([
            Dense(int(encoding_dim * 2), input_shape=(encoding_dim,), activation='relu'),
            Dense(output_dim, activation='sigmoid')  # Final layer to reconstruct the original input
        ])
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, encoded_data, original_data, epochs=50, batch_size=256):
        """
        Trains the decoder model on provided encoded data to reconstruct the original data.

        Args:
            encoded_data (np.array): Encoded data from the encoder.
            original_data (np.array): Original data to reconstruct.
            epochs (int): Number of epochs to train for.
            batch_size (int): Batch size for training.
        """
        self.model.fit(encoded_data, original_data, epochs=epochs, batch_size=batch_size)

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
