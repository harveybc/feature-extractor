from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam

class ExampleEncoderPlugin:
    """
    An encoder plugin using 1D convolutional layers, suitable for time series data,
    with dynamically configurable output size.
    """
    def __init__(self):
        """
        Initializes the ExampleEncoderPlugin without a fixed architecture.
        """
        self.model = None

    def configure_size(self, input_length, input_features, latent_dim):
        """
        Configure the encoder model architecture dynamically based on the size of the input data and the desired latent dimension.

        Args:
            input_length (int): The length of the input sequences.
            input_features (int): The number of features per timestep.
            latent_dim (int): The desired size of the output latent dimension.
        """
        input_layer = Input(shape=(input_length, input_features))
        x = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Flatten()(x)
        x = Dense(latent_dim, activation='relu')(x)  # Adjusts the latent space dimension dynamically

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
