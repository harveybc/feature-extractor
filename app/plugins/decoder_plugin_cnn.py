import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Flatten, Conv1DTranspose,Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal

from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, MaxPooling1D, Cropping1D, LeakyReLU,Input
import math
from tensorflow.keras.layers import ZeroPadding1D

class Plugin:
    plugin_params = {
        'intermediate_layers': 3, 
        'learning_rate': 0.00008,
        'dropout_rate': 0.001,
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

    def configure_size(self, interface_size, output_shape, num_channels, encoder_output_shape):
        print(f"[DEBUG] Starting decoder configuration with interface_size={interface_size}, output_shape={output_shape}, num_channels={num_channels}, encoder_output_shape={encoder_output_shape}")
        
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        # Extract sequence_length and num_filters from encoder_output_shape
        sequence_length, num_filters = encoder_output_shape
        print(f"[DEBUG] Extracted sequence_length={sequence_length}, num_filters={num_filters} from encoder_output_shape.")

        # Calculate the sizes of the intermediate layers based on halving the size
        num_intermediate_layers = self.params['intermediate_layers']
        print(f"[DEBUG] Number of intermediate layers={num_intermediate_layers}")
        
        layers = [output_shape*2]  # Start with output_shape
        current_size = output_shape*2

        # Calculate the layer sizes by halving, similar to the encoder but reversed
        for i in range(num_intermediate_layers):
            next_size = current_size // 2
            if next_size < interface_size:  # Stop if size falls below interface_size
                next_size = interface_size
            layers.append(next_size)
            current_size = next_size

        layers.append(interface_size)  # Ensure the last layer size is the interface size
        print(f"[DEBUG] Calculated decoder layer sizes: {layers}")

        # Reverse the layers for the decoder
        layer_sizes = layers
        print(f"[DEBUG] Decoder layer sizes: {layer_sizes}")

        # Initialize Sequential model for decoder
        self.model = Sequential(name="decoder")
        print(f"[DEBUG] Initialized Sequential model for decoder.")

        # First Conv1DTranspose layer (inverse of the last Conv1D in the encoder)
        print(f"[DEBUG] Adding first Conv1DTranspose layer with input_shape=(sequence_length={sequence_length}, num_filters={num_filters})")
        
        # Adding the first layer (inverse of last layer of encoder)
        self.model.add(Conv1DTranspose(filters=layer_sizes[0],
                                    kernel_size=3,
                                    strides=1,
                                    activation=LeakyReLU(alpha=0.1),
                                    kernel_initializer=HeNormal(),
                                    kernel_regularizer=l2(0.001),
                                    padding='same',
                                    input_shape=(sequence_length, num_filters)))
        print(f"[DEBUG] After first Conv1DTranspose: {self.model.layers[-1].output_shape} [DESIRED: (None, {sequence_length}, {layer_sizes[0]})]")

        self.model.add(BatchNormalization())
        print(f"[DEBUG] After first BatchNormalization: {self.model.layers[-1].output_shape} [DESIRED: (None, {sequence_length}, {layer_sizes[0]})]")

        # Loop through the reversed layers to mirror the encoder's Conv1D layers
        for idx, size in enumerate(layer_sizes[1:], start=1):
            print(f"[DEBUG] Processing layer {idx} with target filter size={size}")

            # If we used strides=2 in the encoder, use strides=2 in reverse Conv1DTranspose to upscale
            strides = 2 if idx < len(layer_sizes) - 1 else 1
            print(f"[DEBUG] Strides set to {strides} for this layer.")

            self.model.add(Conv1DTranspose(filters=size,
                                        kernel_size=3,
                                        strides=strides,  # Reverse of encoder's downsampling
                                        padding='same',
                                        activation=LeakyReLU(alpha=0.1),
                                        kernel_initializer=HeNormal(),
                                        kernel_regularizer=l2(0.001)))
            print(f"[DEBUG] After Conv1DTranspose (filters={size}, strides={strides}): {self.model.layers[-1].output_shape}")

            self.model.add(BatchNormalization())
            print(f"[DEBUG] After BatchNormalization: {self.model.layers[-1].output_shape}")

        # Final Conv1DTranspose layer to match the input dimensions of the encoder (the original number of channels)
        print(f"[DEBUG] Adding final Conv1DTranspose layer to match num_channels={num_channels}")
        self.model.add(Conv1DTranspose(filters=num_channels,
                                    kernel_size=3,
                                    padding='same',
                                    activation=LeakyReLU(alpha=0.1),
                                    kernel_initializer=HeNormal(),
                                    kernel_regularizer=l2(0.001),
                                    name="decoder_output"))
        print(f"[DEBUG] After Final Conv1DTranspose: {self.model.layers[-1].output_shape} [DESIRED FINAL SHAPE: (None, {output_shape}, {num_channels})]")

        # Final Output Shape
        final_output_shape = self.model.layers[-1].output_shape
        print(f"[DEBUG] Final Output Shape: {final_output_shape} [DESIRED FINAL SHAPE: (None, {output_shape}, {num_channels})]")

        # Define the Adam optimizer with custom parameters
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],  # Set the learning rate
            beta_1=0.9,  # Default value
            beta_2=0.999,  # Default value
            epsilon=1e-7,  # Default value
            amsgrad=False  # Default value
        )
        print(f"[DEBUG] Adam optimizer initialized.")

        self.model.compile(optimizer=adam_optimizer, loss='mae')
        print(f"[DEBUG] Model compiled successfully.")




    def train(self, encoded_data, original_data):
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        original_data = original_data.reshape((original_data.shape[0], -1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1, callbacks=[early_stopping])

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
