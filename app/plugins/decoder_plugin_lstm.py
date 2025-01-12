import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

class Plugin:
    """
    A Bidirectional Long Short-Term Memory (Bi-LSTM) network-based decoder using Keras, with dynamically configurable size.
    """

    plugin_params = {
        'intermediate_layers': 3,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'dropout_rate': 0.1,
    }

    plugin_debug_vars = ['interface_size', 'output_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)



    def configure_size(self, interface_size, output_shape, num_channels=None, encoder_output_shape=None, use_sliding_windows=False):
        """
        Configures the decoder model with the specified latent space size and output shape.

        Args:
            interface_size (int): Size of the latent space (input dimensions).
            output_shape (int): Number of time steps in the output sequence.
            num_channels (int, optional): Number of output channels (default: 1 for univariate).
            encoder_output_shape (tuple, optional): Shape of the encoder's output.
            use_sliding_windows (bool, optional): Whether sliding windows are being used (default: False).
        """
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        # Determine input shape
        input_shape = (interface_size,) if not use_sliding_windows else (interface_size, num_channels or 1)

        layer_sizes = []
        current_size = output_shape
        while current_size > interface_size:
            layer_sizes.append(current_size)
            current_size = max(current_size // self.params['layer_size_divisor'], interface_size)
        layer_sizes.append(interface_size)
        layer_sizes.reverse()

        # Debugging message
        print(f"Decoder Layer sizes: {layer_sizes}")

        # Build the model
        self.model = Sequential(name="decoder")

        # Input Dense layer
        self.model.add(Dense(layer_sizes[0], input_shape=input_shape, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01), name="decoder_input"))
        print(f"Added Dense layer with size: {layer_sizes[0]} as decoder_input")

        # RepeatVector layer to match output sequence length
        self.model.add(RepeatVector(output_shape))
        print(f"Added RepeatVector layer with size: {output_shape}")

        # Add Bi-LSTM layers
        for size in layer_sizes:
            self.model.add(Bidirectional(LSTM(units=size, activation='tanh', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.01), return_sequences=True)))
            print(f"Added Bi-LSTM layer with size: {size}")

        # Final TimeDistributed Dense layer to match the output channels
        self.model.add(TimeDistributed(Dense(num_channels or 1, activation='tanh', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.01))))
        print(f"Added TimeDistributed Dense layer with size: {num_channels or 1}")

        # Compile the model
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("Decoder model compiled successfully.")



    def train(self, encoded_data, original_data, validation_data):
        """
        Trains the decoder model with early stopping.

        Args:
            encoded_data (np.ndarray): Encoded input data (latent space representation).
            original_data (np.ndarray): Original target data (ground truth).
            validation_data (tuple): Validation data as (encoded_validation, original_validation).
        """
        print(f"Training decoder with encoded data shape: {encoded_data.shape} and original data shape: {original_data.shape}")
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        original_data = original_data.reshape((original_data.shape[0], original_data.shape[1], 1))

        # Validation data
        encoded_validation, original_validation = validation_data
        encoded_validation = encoded_validation.reshape((encoded_validation.shape[0], -1))
        original_validation = original_validation.reshape((original_validation.shape[0], original_validation.shape[1], 1))

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        self.model.fit(
            encoded_data, original_data,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=(encoded_validation, original_validation),
            callbacks=[early_stopping],
            verbose=1
        )
        print("Training completed.")


    def decode(self, encoded_data, num_time_steps):
        """
        Decodes the latent representation back into the original time series.

        Args:
            encoded_data (np.ndarray): Encoded input data (latent space representation).
            num_time_steps (int): Number of time steps in the output sequence.

        Returns:
            np.ndarray: Decoded output data.
        """
        print(f"Decoding data with shape: {encoded_data.shape}")
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))  # Flatten the data
        decoded_data = self.model.predict(encoded_data)
        decoded_data = decoded_data.reshape((decoded_data.shape[0], num_time_steps, -1))  # Reshape to (batch_size, time_steps, num_channels)
        print(f"Decoded data shape: {decoded_data.shape}")
        return decoded_data


    def save(self, file_path):
        self.model.save(file_path)
        print(f"Decoder model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Decoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], original_data.shape[1], 1))  # Reshape to (batch_size, time_steps, 1)
        reconstructed_data = reconstructed_data.reshape((reconstructed_data.shape[0], reconstructed_data.shape[1], 1))  # Reshape to (batch_size, time_steps, 1)
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"Calculated MSE: {mse}")
        return mse

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=4, output_shape=32)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
