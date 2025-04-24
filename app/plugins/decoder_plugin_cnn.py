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
        # latent shape: window_size // 4  by merged_units
        decoder_input = Input(
            shape=(window_size // 4, branch_units),
            name="decoder_input"
        )
        x= decoder_input

        

        # invert 2nd conv (encoder’s 2nd conv mapped to merged_units)
        # so first deconv should map back to branch_units
        x = Conv1DTranspose(
            filters=branch_units,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=activation,
            name="deconv_branch_units",
            #kernel_regularizer=l2(l2_reg)
        )(x)

        # invert 1st conv (encoder’s 1st conv mapped to branch_units)
        # so next deconv maps back to num_channels
        x = Conv1DTranspose(
            filters=num_channels,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=activation,
            name="deconv_output_channels",
            #kernel_regularizer=l2(l2_reg)
        )(x)

        # if we overshot the exact window_size, crop back
        if x.shape[1] != window_size:
            crop = x.shape[1] - window_size
            x = Cropping1D((0, crop))(x)

        merged = x

        # Output batch normalization layer
        #outputs = BatchNormalization()(x)
        outputs = merged
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
