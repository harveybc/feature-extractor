import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

class DefaultEncoder:
    """
    A simple neural network based encoder using Keras, with dynamically configurable size.
    """
    def __init__(self):
        """
        Initializes the DefaultEncoder without a fixed architecture, to be configured later.
        """
        self.model = None

    def configure_size(self, input_dim, encoding_dim):
        """
        Configure the encoder model architecture based on input and encoding dimensions.

        Args:
            input_dim (int): The number of input features.
            encoding_dim (int): The size of the encoding layer.
        """
        # Initialize the model architecture
        self.model = Sequential([
            Dense(encoding_dim, input_shape=(input_dim,), activation='relu'),
            Dense(int(encoding_dim / 2), activation='relu'),  # Intermediate compression
            Dense(input_dim, activation='sigmoid')  # Output layer to reconstruct the input
        ])
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, data, epochs=50, batch_size=256):
        """
        Trains the encoder model on provided data.

        Args:
            data (np.array): Training data.
            epochs (int): Number of epochs to train for.
            batch_size (int): Batch size for training.
        """
        self.model.fit(data, data, epochs=epochs, batch_size=batch_size)

    def encode(self, data):
        """
        Encodes the data using the trained model.

        Args:
            data (np.array): Data to encode.

        Returns:
            np.array: Encoded data.
        """
        # Assuming the encoding layer is the second layer in the model
        encoder_layer = self.model.layers[1]
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
