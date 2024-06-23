import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Flatten
from keras.optimizers import Adam
from math import ceil

class Plugin:
    """
    A convolutional neural network (CNN) based decoder using Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256,
        'intermediate_layers': 1,
        'layer_size_divisor': 2
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'interface_size', 'output_shape', 'intermediate_layers']

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

        layer_sizes = []
        current_size = output_shape
        layer_size_divisor = self.params['layer_size_divisor'] 
        current_location = output_shape
        int_layers = 0
        while (current_size > interface_size) and (int_layers < (self.params['intermediate_layers']+1)):
            layer_sizes.append(current_location)
            current_size = max(current_size // layer_size_divisor, interface_size)
            current_location = interface_size + current_size
            int_layers += 1
        layer_sizes.append(interface_size)
        layer_sizes.reverse()

        # Debugging message
        print(f"Decoder Layer sizes: {layer_sizes}")

        self.model = Sequential(name="decoder")

        # Adding a single Dense layer
        prev_size = layer_sizes[0]
        self.model.add(Dense(prev_size, input_shape=(interface_size,), activation='relu', name="decoder_input"))
        print(f"Added Dense layer with size: {prev_size} as decoder_input")

        self.model.add(Reshape((prev_size, 1)))
        print(f"Reshape layer with size: ({prev_size}, 1)")
        
        # kernel size configuration based on the layer's size
        kernel_size = 3 
        if prev_size > 64:
            kernel_size = 5
        if prev_size > 512:
            kernel_size = 7

        print(f"Added Conv1D layer with size: {prev_size} and kernel size: {kernel_size}")
        self.model.add(Conv1D(prev_size, kernel_size=kernel_size, padding='same', activation='relu'))
        upsample_factor = 1
        # add intermediate layers
        for i in range(1, len(layer_sizes)):
            prev_size = layer_sizes[i-1]
        
            upsample_factor = ceil(layer_sizes[i] / prev_size)
            
            print(f"Added UpSampling1D layer with upsample factor: {upsample_factor}")
            self.model.add(UpSampling1D(size=upsample_factor))
            
            # kernel size configuration based on the layer's size
            kernel_size = 3 
            if layer_sizes[i] > 64:
                kernel_size = 5
            if layer_sizes[i] > 512:
                kernel_size = 7

            print(f"Added Conv1D layer with size: {layer_sizes[i]} and kernel size: {kernel_size}")
            self.model.add(Conv1D(layer_sizes[i], kernel_size=kernel_size, padding='same', activation='relu'))
        # Adding reshape layer to match the output shape
        self.model.add(Reshape((output_shape, upsample_factor*prev_size)))

        print(f"Added Conv1D layer with size: 1 and kernel size: {kernel_size}")
        self.model.add(Conv1D(1, kernel_size=kernel_size, padding='same', activation='relu'))
        # Adding reshape layer to match the output shape
        self.model.add(Reshape((output_shape,)))

        
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

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
        decoded_data = decoded_data.reshape((decoded_data.shape[0], -1))  # Reshape to 2D array
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
