from keras.models import Model
from keras.layers import Input, LayerNormalization, MultiHeadAttention, Dense, Dropout
from keras.optimizers import Adam

class TransformerEncoderPlugin:
    """
    A Transformer-based encoder plugin with multiple layers, suitable for handling complex sequence data.
    """
    def __init__(self):
        """
        Initializes the TransformerEncoderPlugin without a fixed architecture.
        """
        self.model = None

    def configure_size(self, input_length, input_features, latent_dim):
        """
        Configure the Transformer encoder model architecture dynamically.

        Args:
            input_length (int): The number of timesteps in each input sequence.
            input_features (int): The number of features per timestep.
            latent_dim (int): The desired size of the output latent dimension.
        """
        input_layer = Input(shape=(input_length, input_features))
        x = LayerNormalization()(input_layer)

        # Multiple Transformer layers
        for _ in range(3):  # Configurable number of Transformer layers
            x = MultiHeadAttention(num_heads=2, key_dim=input_features)(x, x)
            x = Dropout(0.1)(x)
            x = LayerNormalization()(x)

        x = Dense(latent_dim, activation='relu')(x)

        self.model = Model(inputs=input_layer, outputs=x)
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, data, epochs=50, batch_size=256):
        """
        Trains the Transformer encoder model on provided data.

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
