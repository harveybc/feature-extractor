import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Flatten, Conv1DTranspose,Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from keras.models import Model, load_model, save_model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, MaxPooling1D, Cropping1D, LeakyReLU,Input
import math
from tensorflow.keras.layers import ZeroPadding1D
import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Lambda, Concatenate
# Add any other necessary imports (like l2 if uncommenting regularizers)

class Plugin:
    plugin_params = {
        'intermediate_layers': 3, 
        'learning_rate': 0.00002,
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

    def configure_size(self, interface_size, output_shape, num_channels, encoder_output_shape, use_sliding_windows, config=None):
        print(f"[DEBUG] Starting decoder configuration with interface_size={interface_size}, output_shape={output_shape}, num_channels={num_channels}, encoder_output_shape={encoder_output_shape}, use_sliding_windows={use_sliding_windows}")
        
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        sequence_length, num_filters = encoder_output_shape
        print(f"[DEBUG] Extracted sequence_length={sequence_length}, num_filters={num_filters} from encoder_output_shape.")

        num_intermediate_layers = self.params['intermediate_layers']
        print(f"[DEBUG] Number of intermediate layers={num_intermediate_layers}")
        
        layers = [output_shape*2]
        current_size = output_shape*2
        l2_reg = 1e-2
        for i in range(num_intermediate_layers-1):
            next_size = current_size // 2
            if next_size < interface_size:
                next_size = interface_size
            layers.append(next_size)
            current_size = next_size

        layers.append(interface_size)

        layer_sizes = layers[::-1]
        print(f"[DEBUG] Calculated decoder layer sizes: {layer_sizes}")

        window_size =config.get("window_size", 288)
        merged_units = config.get("initial_layer_size", 128)
        branch_units = merged_units//config.get("layer_size_divisor", 2)
        activation = config.get("activation", "tanh")
        l2_reg = config.get("l2_reg", self.params.get("l2_reg", 1e-6))

        # --- Decoder input (latent) ---
        # Assume the following variables are defined in the outer scope:
        # window_size, num_channels, feature_units (== merged_units from encoder),
        # num_intermediate_layers, activation, l2_reg (if used)
        feature_units = merged_units


        # --- Calculate Encoder Output Shape ---
        latent_seq_len = feature_units # Time dimension corresponds to feature_units
        latent_feature_dim = num_channels # Channel dimension corresponds to original channels
        encoder_output_shape = (latent_seq_len, latent_feature_dim)
        print(f"[DEBUG] Calculated Encoder Output Shape (Decoder Input): {encoder_output_shape}")

        # --- Decoder input (latent) ---
        decoder_input = Input(
            shape=encoder_output_shape,
            name="decoder_input"
        )

        # --- Feature Decoder: Parallel Isolated Processing Branches ---
        reconstructed_feature_branches = []

        # Split the latent input back into individual channel representations
        # Input shape: (batch, feature_units, num_channels)
        split_inputs = []
        for c in range(num_channels):
            # Lambda layer to slice along the channel axis (axis=2)
            channel_slice = Lambda(
                lambda x, channel=c: x[:, :, channel:channel+1],
                name=f"decoder_split_channel_{c+1}"
            )(decoder_input) # Output shape: (batch, feature_units, 1)
            split_inputs.append(channel_slice)

        # Process each branch to reconstruct the original time series for that channel
        for c in range(num_channels):
            # Current branch input shape: (batch, feature_units, 1)
            x = split_inputs[c]

            # Inverse of Reshape: Flatten the feature_units dimension
            # Input: (batch, feature_units, 1) -> Output: (batch, feature_units)
            x = Flatten(name=f"decoder_feature_{c+1}_flatten")(x)

            # Inverse of Dense Layers
            # Apply Dense layers in reverse order conceptually
            # The final layer must output window_size features
            for i in range(num_intermediate_layers -1): # Apply intermediate layers first
                x = Dense(feature_units, activation=activation, # kernel_regularizer=l2(l2_reg), # Uncomment if used
                        name=f"decoder_feature_{c+1}_dense_{i+1}")(x)
                print(f"[DEBUG] Decoder Branch {c+1} shape after Dense_{i+1}: {K.int_shape(x)}")

            # The last dense layer maps back to the original flattened window size
            x = Dense(window_size, activation='linear', # Use linear activation for reconstruction output
                    # kernel_regularizer=l2(l2_reg), # Uncomment if used
                    name=f"decoder_feature_{c+1}_dense_final_towindow")(x) # Output shape: (batch, window_size)
            print(f"[DEBUG] Decoder Branch {c+1} shape after Dense_final: {K.int_shape(x)}")


            # Inverse of Flatten: Reshape back to (window_size, 1)
            # Input: (batch, window_size) -> Output: (batch, window_size, 1)
            x = Reshape((window_size, 1), name=f"decoder_feature_{c+1}_reshape")(x)
            print(f"[DEBUG] Decoder Branch {c+1} shape after Reshape: {K.int_shape(x)}")

            reconstructed_feature_branches.append(x)

        # --- Concatenate reconstructed branches ---
        # Merge the reconstructed channels (each shape: (batch, window_size, 1))
        # back into a single tensor (batch, window_size, num_channels)
        merged_reconstruction = Concatenate(axis=2, name="decoder_merged_reconstruction")(reconstructed_feature_branches)

        # --- Final Output ---
        decoder_output = merged_reconstruction
        print(f"[DEBUG] Final Decoder Output shape: {K.int_shape(decoder_output)}") # Target: (batch, window_size, num_channels)

        # Output batch normalization layer
        #outputs = BatchNormalization()(x)
        outputs = merged_reconstruction
        print(f"[DEBUG] Final Output shape: {outputs.shape}")

        # Build the encoder model
        self.model = Model(inputs=decoder_input, outputs=outputs, name="decoder")
        
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-2,
            amsgrad=False
        )

        self.model.compile(
            optimizer=adam_optimizer,
            loss=Huber(),
            metrics=['mse', 'mae'],
            run_eagerly=False
        )
        print(f"[DEBUG] Model compiled successfully.")




    def train(self, encoded_data, original_data,config):
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        original_data = original_data.reshape((original_data.shape[0], -1))
        early_stopping = EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1)
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=config.get("batch_size"), verbose=1, callbacks=[early_stopping],validation_split = 0.2)

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
