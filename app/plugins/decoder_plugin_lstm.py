import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, RepeatVector, TimeDistributed
from keras.optimizers import Adam

class Plugin:
    """
    A Bidirectional Long Short-Term Memory (Bi-LSTM) network-based decoder using Keras, with dynamically configurable size.
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
        self.model.add(Dense(layer_sizes[0], input_shape=(interface_size,), activation='relu', name="decoder_input"))
        print(f"Added Dense layer with size: {layer_sizes[0]} as decoder_input")
            
        self.model.add(RepeatVector(output_shape))
        print(f"Added RepeatVector layer with size: {output_shape}")

        # Adding Bi-LSTM layers
        for i in range(0, len(layer_sizes)):
            reshape_size = layer_sizes[i]
            self.model.add(Bidirectional(LSTM(units=reshape_size, activation='tanh', return_sequences=True)))
            print(f"Added Bi-LSTM layer with size: {reshape_size}")

        # Adding the final TimeDistributed Dense layer to match the output shape
        self.model.add(TimeDistributed(Dense(1)))
        print(f"Added TimeDistributed Dense layer with size: 1")

        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, encoded_data, original_data):
        # Debugging message
        print(f"Training decoder with encoded data shape: {encoded_data.shape} and original data shape: {original_data.shape}")
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))  # Flatten the data
        original_data = original_data.reshape((original_data.shape[0], original_data.shape[1], 1))  # Reshape to (batch_size, time_steps, 1)
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)
        print("Training completed.")

    def decode(self, encoded_data):
        # Debugging message
        print(f"Decoding data with shape: {encoded_data.shape}")
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))  # Flatten the data
        decoded_data = self.model.predict(encoded_data)
        decoded_data = decoded_data.reshape((decoded_data.shape[0], decoded_data.shape[1]))  # Reshape to (batch_size, time_steps)
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
