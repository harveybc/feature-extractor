import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Flatten, Conv1DTranspose,Dropout, Input
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
from keras.layers import RepeatVector, Bidirectional, LSTM

class Plugin:
    plugin_params = {
        'learning_rate': 0.00002,
        'dropout_rate': 0.001,
    }

    plugin_debug_vars = ['interface_size_config', 'output_shape_config', 'num_channels_config', 'initial_dense_target_shape']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None
        self.params['interface_size_config'] = None
        self.params['output_shape_config'] = None
        self.params['num_channels_config'] = None
        self.params['initial_dense_target_shape'] = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value 

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, interface_size, output_shape, num_channels, encoder_shape_before_flatten, use_sliding_windows, config=None):
        print(f"[DEBUG VAE Decoder] Starting decoder configuration with interface_size(latent_dim)={interface_size}, output_shape(window_size)={output_shape}, num_channels={num_channels}, encoder_shape_before_flatten={encoder_shape_before_flatten}")
        
        self.params['interface_size_config'] = interface_size
        self.params['output_shape_config'] = output_shape
        self.params['num_channels_config'] = num_channels
        self.params['initial_dense_target_shape'] = encoder_shape_before_flatten

        window_size = output_shape
        
        merged_units = config.get("initial_layer_size", 64)
        branch_units = merged_units // config.get("layer_size_divisor", 2)
        lstm_units = branch_units // config.get("layer_size_divisor", 2) 
        l2_reg = config.get("l2_reg", self.params.get("l2_reg", 1e-6))
        
        activation = config.get("activation", "tanh")

        decoder_z_input = Input(shape=(interface_size,), name="decoder_z_input_vae")
        
        dense_units_target = np.prod(encoder_shape_before_flatten)
        
        x = Dense(dense_units_target, activation=activation, name="decoder_initial_dense_vae")(decoder_z_input)
        x = Reshape(encoder_shape_before_flatten, name="decoder_initial_reshape_vae")(x)
        print(f"[DEBUG VAE Decoder] Shape after initial Dense and Reshape: {x.shape}")

        x = Conv1DTranspose(
            filters=lstm_units,
            kernel_size=3, strides=2, padding='same', activation=activation, name="deconv1d_0_vae",
            #kernel_regularizer=l2(l2_reg)
        
        )(x)
        print(f"[DEBUG VAE Decoder] Shape after deconv1d_0: {x.shape}")
        
        x = Conv1DTranspose(
            filters=branch_units,
            kernel_size=3, strides=2, padding='same', activation=activation, name="deconv1d_1_vae",
        #    kernel_regularizer=l2(l2_reg)
        
        )(x)
        print(f"[DEBUG VAE Decoder] Shape after deconv1d_1: {x.shape}")

        x = Conv1DTranspose(
            filters=merged_units,
            kernel_size=3, strides=2, padding='same', activation=activation, name="deconv1d_2_vae",
        #    kernel_regularizer=l2(l2_reg)
        
        )(x)
        print(f"[DEBUG VAE Decoder] Shape after deconv1d_2: {x.shape}")

        x = Conv1DTranspose(
            filters=num_channels,
            kernel_size=3, strides=2, padding='same', activation='linear', name="deconv1d_final_vae",
        #    kernel_regularizer=l2(l2_reg)
        
        )(x)
        print(f"[DEBUG VAE Decoder] Pre-cropping/padding shape: {x.shape}")

        current_seq_len = x.shape[1]
        if current_seq_len is not None and current_seq_len != window_size:
            if current_seq_len > window_size:
                crop_amount = current_seq_len - window_size
                x = Cropping1D((0, crop_amount), name="decoder_cropping_vae")(x)
                print(f"[DEBUG VAE Decoder] Cropped to: {x.shape}")
            elif current_seq_len < window_size:
                padding_amount = window_size - current_seq_len
                x = ZeroPadding1D(padding=(0, padding_amount), name="decoder_padding_vae")(x)
                print(f"[DEBUG VAE Decoder] Padded to: {x.shape}")
        
        outputs = x
        print(f"[DEBUG VAE Decoder] Final Output shape: {outputs.shape}")

        self.model = Model(inputs=decoder_z_input, outputs=outputs, name="vae_decoder")
        
        print(f"[DEBUG VAE Decoder] VAE Decoder model built. Not compiled standalone.")

    def train(self, encoded_data, original_data,config):
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        original_data = original_data.reshape((original_data.shape[0], -1))
        early_stopping = EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1)
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=config.get("batch_size"), verbose=1, callbacks=[early_stopping],validation_split = 0.2)

    def decode(self, encoded_z_data, use_sliding_windows=None, original_feature_size=None):
        if not self.model:
            raise ValueError("Decoder model is not configured or loaded.")
        print(f"[VAE Decoder decode] Decoding data (z) with shape: {encoded_z_data.shape}")
        decoded_data = self.model.predict(encoded_z_data)
        print(f"[VAE Decoder decode] Decoded data raw shape: {decoded_data.shape}")
        return decoded_data

    def save(self, file_path):
        if self.model:
            self.model.save(file_path)
            print(f"VAE Decoder model saved to {file_path}")
        else:
            print("VAE Decoder model not available to save.")

    def load(self, file_path):
        self.model = load_model(file_path, compile=False)
        print(f"VAE Decoder model loaded from {file_path}")
        try:
            self.params['interface_size_config'] = self.model.input_shape[-1]
            self.params['output_shape_config'] = self.model.output_shape[1]
            self.params['num_channels_config'] = self.model.output_shape[-1]
            print(f"[DEBUG VAE Decoder Load] Reconstructed interface_size: {self.params['interface_size_config']}, output_shape: {self.params['output_shape_config']}, num_channels: {self.params['num_channels_config']}")
        except Exception as e:
            print(f"[WARNING VAE Decoder Load] Could not fully reconstruct config params from loaded model. Error: {e}")

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
