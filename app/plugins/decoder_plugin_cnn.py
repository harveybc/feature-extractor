import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Flatten, Conv1DTranspose,Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2

from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, MaxPooling1D, Cropping1D, LeakyReLU,Input
import math
from tensorflow.keras.layers import ZeroPadding1D

class Plugin:
    plugin_params = {
        'intermediate_layers': 3, 
        'learning_rate': 0.001,
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

    def configure_size(self, interface_size, output_shape, num_channels, encoder_output_shape, use_sliding_windows):
        print(f"[DEBUG] Starting decoder configuration with interface_size={interface_size}, output_shape={output_shape}, num_channels={num_channels}, encoder_output_shape={encoder_output_shape}, use_sliding_windows={use_sliding_windows}")
        
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        sequence_length, num_filters = encoder_output_shape
        print(f"[DEBUG] Extracted sequence_length={sequence_length}, num_filters={num_filters} from encoder_output_shape.")

        num_intermediate_layers = self.params['intermediate_layers']
        print(f"[DEBUG] Number of intermediate layers={num_intermediate_layers}")
        
        layers = [output_shape*2]
        current_size = output_shape*2

        for i in range(num_intermediate_layers-1):
            next_size = current_size // 2
            if next_size < interface_size:
                next_size = interface_size
            layers.append(next_size)
            current_size = next_size

        layers.append(interface_size)

        layer_sizes = layers[::-1]
        print(f"[DEBUG] Calculated decoder layer sizes: {layer_sizes}")

        self.model = Sequential(name="decoder")

        print(f"[DEBUG] Adding first Conv1DTranspose layer with input_shape=(sequence_length={sequence_length}, num_filters={num_filters})")
        self.model.add(Conv1DTranspose(
            filters=layer_sizes[0],
            kernel_size=3,
            strides=1,
            activation=LeakyReLU(alpha=0.1),
            kernel_initializer=HeNormal(),
            kernel_regularizer=l2(0.001),
            padding='same',
            input_shape=(sequence_length, num_filters)
        ))
        self.model.add(BatchNormalization())

        for idx, size in enumerate(layer_sizes[1:], start=1):
            strides = 2 if idx < len(layer_sizes) - 1 else 1
            self.model.add(Conv1DTranspose(
                filters=size,
                kernel_size=3,
                strides=strides,
                padding='same',
                activation=LeakyReLU(alpha=0.1),
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(0.001)
            ))
            self.model.add(BatchNormalization())

        final_output_channels = num_channels if use_sliding_windows else output_shape
        self.model.add(Conv1DTranspose(
            filters=final_output_channels,
            kernel_size=3,
            padding='same',
            activation=LeakyReLU(alpha=0.1),
            kernel_initializer=HeNormal(),
            kernel_regularizer=l2(0.001),
            name="decoder_output"
        ))

        final_output_shape = self.model.layers[-1].output_shape
        print(f"[DEBUG] Final Output Shape: {final_output_shape}")

        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )
        self.model.compile(
            optimizer=adam_optimizer,
            loss=Huber(),
            metrics=['mse', 'mae'],
            run_eagerly=False
        )
        print(f"[DEBUG] Model compiled successfully.")





    def train(self, encoded_data, original_data):
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        original_data = original_data.reshape((original_data.shape[0], -1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1, callbacks=[early_stopping])

    def decode(self, encoded_data, use_sliding_windows, original_feature_size):
        print(f"[decode] Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data)

        if not use_sliding_windows:
            # Reshape to match the original feature size if not using sliding windows
            decoded_data = decoded_data.reshape((decoded_data.shape[0], original_feature_size))
            print(f"[decode] Reshaped decoded data to match original feature size: {decoded_data.shape}")
        else:
            print(f"[decode] Decoded data shape: {decoded_data.shape}")

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
