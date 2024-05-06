from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam

class LSTMEncoderPlugin:
    """
    An LSTM-based encoder plugin suitable for time series data,
    with dynamically configurable output size.
    """
    def __init__(self):
        """
        Initializes the LSTMEncoderPlugin without a fixed architecture.
        """
        self.model = None

    def configure_size(self, input_length, input_features, latent_dim):
        """
        Configure the encoder model architecture dynamically based on the desired output size.

        Args:
            input_length (int): The number of timesteps in each input sequence.
            input_features (int): The number of features per timestep.
            latent_dim (int): The desired size of the output latent dimension.
        """
        input_layer = Input(shape=(input_length, input_features))
        x = LSTM(64, return_sequences=True)(input_layer)
        x = LSTM(32, return_sequences=False)(x)
        x = Dense(latent_dim, activation='relu')(x)  # Controls the size of the output encoding

        self.model = Model(inputs=input_layer, outputs=x)
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
        self.model.load_weights(file_path)
