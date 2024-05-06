from keras.models import Sequential, Model
from keras.layers import Input, Conv1D, UpSampling1D

class ExampleDecoderPlugin:
    """
    An example decoder plugin using 1D convolutional layers, suitable for time series data,
    with dynamically configurable input size.
    """
    def __init__(self):
        """
        Initializes the ExampleDecoderPlugin without a fixed architecture.
        The model will be created dynamically based on the input configuration.
        """
        self.model = None

    def configure_size(self, input_length, input_filters):
        """
        Configure the decoder model architecture dynamically based on the size of the encoded output.

        Args:
            input_length (int): The length of the input sequences.
            input_filters (int): The number of filters (or features) in the input encoded data.
        """
        input_layer = Input(shape=(input_length, input_filters))
        x = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
        x = UpSampling1D(2)(x)
        x = Conv1D(16, 3, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        output_layer = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

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
        self.model.load_weights(file_path)
