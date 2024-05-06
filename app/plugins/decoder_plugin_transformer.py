from keras.models import Model
from keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, TimeDistributed
from keras.optimizers import Adam

class TransformerDecoderPlugin:
    """
    A Transformer-based decoder plugin suitable for reconstructing time series data from encoded states.
    """
    def __init__(self):
        """
        Initializes the TransformerDecoderPlugin without a fixed architecture.
        """
        self.model = None

    def configure_size(self, input_length, latent_dim, output_features):
        """
        Configure the decoder model architecture dynamically based on the size of the encoded output.

        Args:
            input_length (int): The number of timesteps in the output sequence.
            latent_dim (int): The size of the input latent dimension.
            output_features (int): The number of features per timestep in the output sequence.
        """
        input_layer = Input(shape=(input_length, latent_dim))
        x = LayerNormalization()(input_layer)

        # Reversing the encoder Transformer layers
        for _ in range(3):  # Matching number of Transformer layers as in the encoder
            x = MultiHeadAttention(num_heads=2, key_dim=latent_dim)(x, x)
            x = LayerNormalization()(x)

        x = TimeDistributed(Dense(output_features, activation='sigmoid'))(x)

        self.model = Model(inputs=input_layer, outputs=x)
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
        self.model.load_weights(file_path)
