import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


class Plugin:
    """
    A strictly 'inverse' decoder plugin that mirrors the encoder layer sizes in reverse.
    For example, if encoder = [LSTM(64), LSTM(32), Dense(16)],
    then decoder = [Dense(32), Dense(64), RepeatVector(36), LSTM(64), LSTM(32), TimeDistributed(1)].
    
    Adjust 'inverted_sizes' below to mirror your encoder exactly.
    """

    plugin_params = {
        # Training hyperparameters
        'epochs': 200,
        'batch_size': 128,

        # You can still keep some dropout / L2 if desired:
        'dropout_rate': 0.1,
        'l2_reg': 1e-4,

        # Learning rate
        'learning_rate': 0.001,
    }

    # For debugging/tracking
    plugin_debug_vars = ['latent_dim', 'output_steps', 'output_channels']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def configure_size(self, latent_dim, output_steps, output_channels=1):
        """
        Configures a strictly inverse decoder with a known set of layer sizes:
        
        1) Dense(32) -> Dense(64)
        2) RepeatVector(output_steps=36)
        3) LSTM(64, return_sequences=True)
        4) LSTM(32, return_sequences=True)
        5) TimeDistributed(Dense(output_channels=1))

        This is the 'mirror' of an encoder that had LSTM(64)->LSTM(32)->Dense(16) going from 36 timesteps to 16 latent.
        
        Args:
            latent_dim (int): Dimension of the latent vector output by the encoder (e.g., 16).
            output_steps (int): Number of time steps to reconstruct (e.g., 36).
            output_channels (int): Dimensionality (features) per time step (e.g., 1).
        """

        self.params['latent_dim'] = latent_dim
        self.params['output_steps'] = output_steps
        self.params['output_channels'] = output_channels

        dropout_rate = self.params['dropout_rate']
        l2_reg = self.params['l2_reg']

        print(f"[configure_size] Building inverse decoder for latent_dim={latent_dim}, output={output_steps}x{output_channels}")

        model = Sequential(name="inverse_decoder")

        # -- 1) Dense(32) -> Dense(64) from latent_dim(16) to bigger hidden dims
        # Input shape = (latent_dim,)
        model.add(Dense(
            32,
            input_shape=(latent_dim,),
            activation='relu',
            kernel_initializer=HeNormal(),
            kernel_regularizer=l2(l2_reg),
            name="decoder_dense_32",
        ))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))

        model.add(Dense(
            64,
            activation='relu',
            kernel_initializer=HeNormal(),
            kernel_regularizer=l2(l2_reg),
            name="decoder_dense_64",
        ))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))

        # -- 2) RepeatVector(36) to create 36 time steps
        model.add(RepeatVector(output_steps, name="repeat_vector"))

        # -- 3) LSTM(64, return_sequences=True)
        model.add(LSTM(
            64,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            return_sequences=True,
            name="decoder_lstm_64",
        ))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))

        # -- 4) LSTM(32, return_sequences=True)
        model.add(LSTM(
            32,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            return_sequences=True,
            name="decoder_lstm_32",
        ))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))

        # -- 5) TimeDistributed(Dense(output_channels=1)) => final reconstruction
        model.add(TimeDistributed(
            Dense(
                output_channels,
                activation='tanh',
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg),
            ),
            name="decoder_output"
        ))

        # -- Compile
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        model.summary()

        self.model = model

    def train(self, encoded_data, original_data):
        """
        Train the inverse decoder with (encoded_data -> original_data).

        - encoded_data shape: (batch_size, latent_dim)
        - original_data shape: (batch_size, output_steps, output_channels)
        """
        if self.model is None:
            raise ValueError("[train] Decoder model is not configured. Call configure_size first.")

        print(f"[train] Training with encoded_data shape={encoded_data.shape}, original_data shape={original_data.shape}")
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']

        # If original_data is (batch_size, output_steps) for univariate, reshape => (batch_size, output_steps, 1)
        if len(original_data.shape) == 2:
            original_data = np.expand_dims(original_data, axis=-1)

        # Potentially add early stopping
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            encoded_data, original_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stopping]
        )
        print("[train] Training completed.")
        return history

    def decode(self, encoded_data):
        """
        Decode latent vectors back into time-series sequences.
        
        Returns shape: (batch_size, output_steps, output_channels).
                       If output_channels=1, final shape => (batch_size, output_steps).
        """
        if self.model is None:
            raise ValueError("[decode] Decoder model is not configured.")

        print(f"[decode] Decoding data with shape={encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data)
        print(f"[decode] Decoded data shape={decoded_data.shape}")

        # If univariate => (batch_size, time_steps, 1), we can squeeze last dimension
        if decoded_data.shape[-1] == 1:
            decoded_data = decoded_data.squeeze(axis=-1)
        return decoded_data

    def calculate_mse(self, original_data, reconstructed_data):
        """
        Calculate MSE between original_data and reconstructed_data.
        Expects shapes => (batch_size, time_steps, 1) for univariate or (batch_size, time_steps, channels).
        """
        if len(original_data.shape) == 2:  # (batch_size, time_steps)
            original_data = np.expand_dims(original_data, axis=-1)
        if len(reconstructed_data.shape) == 2:  # (batch_size, time_steps)
            reconstructed_data = np.expand_dims(reconstructed_data, axis=-1)

        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"[calculate_mse] MSE={mse}")
        return mse

    def save(self, file_path):
        """
        Save the decoder model.
        """
        if self.model is None:
            raise ValueError("[save] Decoder model is not configured.")
        self.model.save(file_path)
        print(f"[save] Decoder model saved to {file_path}")

    def load(self, file_path):
        """
        Load a pre-trained decoder model from file.
        """
        self.model = load_model(file_path)
        print(f"[load] Decoder model loaded from {file_path}")


# Example usage
if __name__ == "__main__":
    # Suppose your encoder reduced shape (36,1) -> 16D latent
    # Then this decoder expands 16D latent -> shape (36,1).
    plugin = Plugin()
    plugin.configure_size(latent_dim=16, output_steps=36, output_channels=1)

    # Generate synthetic data
    batch_size = 8
    # Latent data: (batch_size, 16)
    encoded_data = np.random.randn(batch_size, 16)
    # Original data to reconstruct: (batch_size, 36, 1) => univariate
    original_data = np.random.randn(batch_size, 36)

    # Train
    plugin.train(encoded_data, original_data)

    # Decode
    decoded = plugin.decode(encoded_data)
    print(f"Decoded shape: {decoded.shape}")  
