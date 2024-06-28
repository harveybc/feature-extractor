import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Flatten, Conv1DTranspose
from keras.optimizers import Adam

class Plugin:
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

        self.model = Sequential(name="decoder")
        self.model.add(Dense(layer_sizes[0], input_shape=(interface_size,), activation='relu', name="decoder_input"))
        self.model.add(Reshape((layer_sizes[0], 1)))
        reshape_size = layer_sizes[0]
        next_size = layer_sizes[0]
        for i in range(0, len(layer_sizes) - 1):
            kernel_size = 3
            if layer_sizes[i] > 64:
                kernel_size = 5
            if layer_sizes[i] > 512:
                kernel_size = 7
            self.model.add(Conv1DTranspose(next_size, kernel_size=kernel_size, padding='same', activation='relu'))
            reshape_size = layer_sizes[i]
            next_size = layer_sizes[i + 1]
            upsample_factor = next_size // reshape_size
            if upsample_factor > 1:
                self.model.add(UpSampling1D(size=upsample_factor))
        self.model.add(Conv1DTranspose(output_shape, kernel_size=kernel_size, padding='same', activation='tanh', name="last_layer"))
        last_layer_shape = self.model.layers[-1].output_shape
        new_shape = (last_layer_shape[2], last_layer_shape[1])
        self.model.add(Reshape(new_shape))
        self.model.add(Conv1DTranspose(1, kernel_size=3, padding='same', activation='tanh', name="decoder_output"))
        self.model.add(Reshape((output_shape,)))
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, encoded_data, original_data):
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        original_data = original_data.reshape((original_data.shape[0], -1))
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)

    def decode(self, encoded_data):
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        decoded_data = self.model.predict(encoded_data)
        decoded_data = decoded_data.reshape((decoded_data.shape[0], -1))
        return decoded_data

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = load_model(file_path)

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], -1))
        reconstructed_data = reconstructed_data.reshape((original_data.shape[0], -1))
        mse = np.mean(np.square(original_data - reconstructed_data))
        return mse

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=4, output_shape=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
