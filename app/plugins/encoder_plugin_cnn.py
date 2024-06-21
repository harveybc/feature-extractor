import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, Flatten, Dense, Input
from keras.optimizers import Adam

class Plugin:
    """
    A CNN-based encoder plugin using Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'encoding_dim']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, input_dim, encoding_dim):
        self.params['input_dim'] = input_dim
        self.params['encoding_dim'] = encoding_dim

        # Encoder
        encoder_input = Input(shape=(input_dim, 1), name="encoder_input")
        x = Conv1D(filters=16, kernel_size=3, activation='relu')(encoder_input)
        x = Flatten()(x)
        encoder_output = Dense(encoding_dim, activation='relu', name="encoder_output")(x)
        self.encoder_model = Model(inputs=encoder_input, outputs=encoder_output, name="encoder")
        self.encoder_model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Debugging messages to trace the model configuration
        print("Encoder Model Summary:")
        self.encoder_model.summary()

    def train(self, data):
        print(f"Training encoder with data shape: {data.shape}")
        data = np.expand_dims(data, axis=-1)  # Add a channel dimension
        self.encoder_model.fit(data, data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)
        print("Training completed.")

    def encode(self, data):
        print(f"Encoding data with shape: {data.shape}")
        data = np.expand_dims(data, axis=-1)  # Add a channel dimension
        encoded_data = self.encoder_model.predict(data)
        print(f"Encoded data shape: {encoded_data.shape}")
        return encoded_data

    def save(self, file_path):
        save_model(self.encoder_model, file_path)
        print(f"Encoder model saved to {file_path}")

    def load(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"Encoder model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(input_dim=128, encoding_dim=4)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
