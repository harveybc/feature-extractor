import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Flatten, Conv1DTranspose,Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal

from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, MaxPooling1D, Cropping1D, LeakyReLU,Input
import math

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

        layer_sizes = []

        # Calculate the sizes of the intermediate layers
        num_intermediate_layers = self.params['intermediate_layers']
        print(f"[DEBUG] Number of intermediate layers={num_intermediate_layers}")
        
        layers = [output_shape]
        step_size = (output_shape - interface_size) / (num_intermediate_layers + 1)
        print(f"[DEBUG] Calculated step_size={step_size}")

        for i in range(1, num_intermediate_layers + 1):
            layer_size = output_shape - i * step_size
            print(f"[DEBUG] Calculated layer_size for layer {i}: {layer_size}")
            layers.append(int(layer_size))

        layers.append(interface_size)
        print(f"[DEBUG] Appended interface_size={interface_size} to layers: {layers}")

        # Reverse the layers for the decoder
        layer_sizes = layers
        layer_sizes.reverse()
        print(f"[DEBUG] Reversed layer sizes: {layer_sizes} [DESIRED FINAL SHAPE: (None, 128, {num_channels})]")

        # Initialize Sequential model
        self.model = Sequential(name="decoder")
        print(f"[DEBUG] Initialized Sequential model for decoder.")

        # First Conv1DTranspose layer
        print(f"[DEBUG] Adding first Conv1DTranspose layer with input_shape=(sequence_length={sequence_length}, num_filters={num_filters})")
        self.model.add(Conv1DTranspose(filters=layer_sizes[0],
                                    kernel_size=3,
                                    strides=1,
                                    activation=LeakyReLU(alpha=0.1),
                                    kernel_initializer=HeNormal(),
                                    kernel_regularizer=l2(0.001),
                                    padding='same',
                                    input_shape=(sequence_length, num_filters)))
        print(f"[DEBUG] After first Conv1DTranspose: {self.model.layers[-1].output_shape} [DESIRED: (None, 53, {layer_sizes[0]})]")

        self.model.add(BatchNormalization())
        print(f"[DEBUG] After first BatchNormalization: {self.model.layers[-1].output_shape} [DESIRED: (None, 53, {layer_sizes[0]})]")

        # Explicit tracking of sequence length
        current_sequence_length = sequence_length
        print(f"[DEBUG] Current sequence length after first Conv1DTranspose: {current_sequence_length}")

        # Upsampling loop
        for idx, size in enumerate(layer_sizes[1:], start=1):
            print(f"[DEBUG] Processing layer {idx} with target filter size={size}")
            last_shape = self.model.layers[-1].output_shape
            current_sequence_length = int(last_shape[1])
            print(f"[DEBUG] Current sequence length before upsampling: {current_sequence_length}")

            # Check for upsampling requirement
            if current_sequence_length < self.params['output_shape']:
                strides = 2 if current_sequence_length * 2 <= self.params['output_shape'] else 1
                print(f"[DEBUG] Strides set to {strides} for this layer (current_sequence_length={current_sequence_length})")
            else:
                strides = 1
                print(f"[DEBUG] No upsampling required, strides set to {strides}")

            # Manually track the new sequence length
            new_sequence_length = current_sequence_length * strides
            print(f"[DEBUG] New sequence length after applying strides: {new_sequence_length}")

            self.model.add(Conv1DTranspose(filters=size,
                                        kernel_size=3,
                                        strides=strides,
                                        padding='same',
                                        activation=LeakyReLU(alpha=0.1),
                                        kernel_initializer=HeNormal(),
                                        kernel_regularizer=l2(0.001)))
            print(f"[DEBUG] After Conv1DTranspose (filters={size}, strides={strides}): {self.model.layers[-1].output_shape} [DESIRED: (None, {new_sequence_length}, {size})]")

            self.model.add(BatchNormalization())
            print(f"[DEBUG] After BatchNormalization: {self.model.layers[-1].output_shape} [DESIRED: (None, {new_sequence_length}, {size})]")

        # Final padding calculation
        actual_sequence_length = current_sequence_length
        required_padding = 128 - actual_sequence_length  # Calculate the padding needed to reach 128
        print(f"[DEBUG] Calculated required padding to reach 128: {required_padding}")
        
        padding_left = required_padding // 2
        padding_right = required_padding - padding_left
        print(f"[DEBUG] Applying ZeroPadding1D with padding_left={padding_left}, padding_right={padding_right}")

        # Apply padding before the final Conv1DTranspose layer
        self.model.add(ZeroPadding1D(padding=(padding_left, padding_right)))
        print(f"[DEBUG] After ZeroPadding1D: {self.model.layers[-1].output_shape} [DESIRED: (None, 128, {layer_sizes[-1]})]")

        # Final Conv1DTranspose layer to match the original number of channels
        print(f"[DEBUG] Adding final Conv1DTranspose layer to match num_channels={num_channels}")
        self.model.add(Conv1DTranspose(filters=num_channels,
                                    kernel_size=3,
                                    padding='same',
                                    activation=LeakyReLU(alpha=0.1),
                                    kernel_initializer=HeNormal(),
                                    kernel_regularizer=l2(0.001),
                                    name="decoder_output"))
        print(f"[DEBUG] After Final Conv1DTranspose: {self.model.layers[-1].output_shape} [DESIRED FINAL SHAPE: (None, 128, {num_channels})]")

        # Final Output Shape
        final_output_shape = self.model.layers[-1].output_shape
        print(f"[DEBUG] Final Output Shape: {final_output_shape} [DESIRED FINAL SHAPE: (None, 128, {num_channels})]")

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
