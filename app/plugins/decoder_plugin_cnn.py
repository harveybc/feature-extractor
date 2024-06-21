import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Flatten
from keras.optimizers import Adam

class Plugin:
    """
    A convolutional neural network (CNN) based decoder using Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'interface_size', 'output_shape']

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

    def configure_size(self, interface_size, output_shape):
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        # Calculate layer sizes starting from the output size
        layer_sizes = []
        current_size = output_shape
        while current_size > interface_size:
            layer_sizes.append(current_size)
            current_size = max(current_size // 4, interface_size)
        layer_sizes.append(interface_size)
        layer_sizes.reverse()

        print(f"Layer sizes: {layer_sizes}")

        self.model = Sequential(name="decoder")
        
        # Start with dense layer of interface size
        self.model.add(Dense(interface_size, input_shape=(interface_size,), activation='relu', name="decoder_input"))

        # Add Dense layers and reshape
        for i in range(1, len(layer_sizes)):
            self.model.add(Dense(layer_sizes[i], activation='relu'))
            if i < len(layer_sizes) - 1:
                reshape_size = layer_sizes[i]
                total_elements = reshape_size * layer_sizes[i + 1] // 4  # Correct total_elements calculation
                self.model.add(Reshape((reshape_size, total_elements // reshape_size)))
                upsampling_factor = layer_sizes[i+1] // layer_sizes[i]
                self.model.add(UpSampling1D(size=upsampling_factor))
                kernel_size = min(3, reshape_size)  # Dynamically set kernel size
                self.model.add(Conv1D(layer_sizes[i+1], kernel_size=kernel_size, padding='same', activation='relu'))

        # Final Convolution layer to match the output shape
        self.model.add(Conv1D(1, kernel_size=min(3, layer_sizes[-1]), padding='same', activation='tanh', name="decoder_output"))
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Debugging messages to trace the model configuration
        print("Decoder Model Summary:")
        self.model.summary()

    def train(self, encoded_data, original_data):
        # Debugging message
        print(f"Training decoder with encoded data shape: {encoded_data.shape} and original data shape: {original_data.shape}")
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))  # Flatten the data
        original_data = original_data.reshape((original_data.shape[0], -1))  # Flatten the data
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)
        print("Training completed.")

    def decode(self, encoded_data):
        # Debugging message
        print(f"Decoding data with shape: {encoded_data.shape}")
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))  # Flatten the data
        decoded_data = self.model.predict(encoded_data)
        print(f"Decoded data shape: {decoded_data.shape}")
        return decoded_data

    def save(self, file_path):
        self.model.save(file_path)
        print(f"Decoder model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Decoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], -1))  # Flatten the data
        reconstructed_data = reconstructed_data.reshape((original_data.shape[0], -1))  # Flatten the data
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"Calculated MSE: {mse}")
        return mse

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=4, output_shape=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
