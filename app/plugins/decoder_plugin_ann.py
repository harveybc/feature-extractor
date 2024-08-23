import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout,Input, LeakyReLU,BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2

class Plugin:
    """
    A simple neural network-based decoder using Keras, with dynamically configurable size.
    """

    plugin_params = {
        'intermediate_layers': 2,
        'learning_rate': 0.0001
    }

    plugin_debug_vars = []

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

    def configure_size(self, encoding_dim, output_dim):
        self.params['encoding_dim'] = encoding_dim
        self.params['output_dim'] = output_dim

        layer_sizes = []
        output_shape = output_dim
        interface_size = encoding_dim
        # Calculate the sizes of the intermediate layers
        num_intermediate_layers = self.params['intermediate_layers']
        layers = [output_shape]
        step_size = (output_shape - interface_size) / (num_intermediate_layers + 1)
        
        for i in range(1, num_intermediate_layers + 1):
            layer_size = output_shape - i * step_size
            layers.append(int(layer_size))

        layers.append(interface_size)

        # For the decoder, reverses the order of the generted layers.
        layer_sizes=layers
        layer_sizes.reverse()
        
        # Debugging message
        print(f"Decoder Layer sizes: {layer_sizes}")

        self.model = Sequential(name="decoder_ANN")
        self.model.add(Input(shape=(encoding_dim), name="decoder_input"))
        #self.model.add(Dense(layer_sizes[0], input_shape=(encoding_dim,), activation='tanh', kernel_initializer=GlorotUniform(), name="decoder_input"))
        layers_index = 0
        for size in layer_sizes[1:-1]:
            layers_index += 1
            self.model.add(Dense(size, activation=LeakyReLU(alpha=0.1), kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001), name="decoder_intermediate_layer" + str(layers_index) ))
            # batch normalization layer
            self.model.add(BatchNormalization(name="decoder_batch_norm" + str(layers_index)))


        self.model.add(Dense(output_dim, activation=LeakyReLU(alpha=0.1), kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001), name="decoder_output"))

        # Define the Adam optimizer with custom parameters
        adam_optimizer = Adam(
            learning_rate= self.params['learning_rate'],   # Set the learning rate
            beta_1=0.9,            # Default value
            beta_2=0.999,          # Default value
            epsilon=1e-7,          # Default value
            amsgrad=False
        )

        self.model.compile(optimizer=adam_optimizer, loss='mae')

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
    plugin.configure_size(encoding_dim=4, output_dim=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
