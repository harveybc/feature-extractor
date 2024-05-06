import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class DefaultEncoder:
    """
    A simple neural network based encoder using Keras.
    """
    def __init__(self, input_dim=100, encoding_dim=50):
        """
        Initializes the DefaultEncoder with a simple feedforward architecture.

        Args:
            input_dim (int): The number of input features.
            encoding_dim (int): The size of the encoding layer.
        """
        self.model = Sequential([
            Dense(encoding_dim, input_shape=(input_dim,), activation='relu'),
            Dense(int(encoding_dim / 2), activation='relu'),  # Further compression
            Dense(encoding_dim, activation='relu')  # Output layer
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

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
        return self.model.predict(data)

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
        from keras.models import load_model
        self.model = load_model(file_path)
