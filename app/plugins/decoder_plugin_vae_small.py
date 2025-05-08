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
from keras.layers import RepeatVector, Bidirectional, LSTM

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

        num_filters = encoder_output_shape[0]
        sequence_length = 18  
        print(f"[DEBUG] Extracted sequence_length={sequence_length}, num_filters={num_filters} from encoder_output_shape.")

        window_size = config.get("window_size", 288)
        merged_units = config.get("initial_layer_size", 128)
        branch_units = merged_units // config.get("layer_size_divisor", 2)
        lstm_units = branch_units // config.get("layer_size_divisor", 2)  # Match LSTM size in encoder
        activation = config.get("activation", "tanh")
        l2_reg = config.get("l2_reg", self.params.get("l2_reg", 1e-6))

        # --- Decoder input (latent) ---
        decoder_input = Input(
            shape=encoder_output_shape,
            name="decoder_input"
        )
        x = decoder_input

        # --- Reverse LSTM ---
        #x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bidir_lstm_reverse")(x)

        # --- Reverse Conv1D Layers ---
        x = Conv1DTranspose(
            filters=lstm_units//2,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=activation,
            name="deconv1d_0",
            #kernel_regularizer=l2(l2_reg)
        )(x)
        
        # --- Reverse Conv1D Layers ---
        x = Conv1DTranspose(
            filters=lstm_units,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=activation,
            name="deconv1d_1"
        )(x)

        # --- Reverse Conv1D Layers ---
        x = Conv1DTranspose(
            filters=branch_units,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=activation,
            name="deconv1d_2"
        )(x)

        x = Conv1DTranspose(
            filters=num_channels,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='linear',
            name="deconv1d_4"
        )(x)
        # print pre cropping shape
        print(f"[DEBUG] Pre-cropping shape: {x.shape}")

        # If we overshot the exact window_size, crop back
        if x.shape[1] != window_size:
            crop = x.shape[1] - window_size
            if crop > 0:
                x = Cropping1D((0, crop))(x)
            else:
                print(f"[DEBUG] Skipping cropping as crop={crop} is negative.")


        merged = x

        # Output batch normalization layer
        outputs = merged
        print(f"[DEBUG] Final Output shape: {outputs.shape}")

        # Build the decoder model
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
