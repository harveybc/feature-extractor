import numpy as np
from keras.models import Sequential, load_model, save_model, Model
from keras.layers import Dense, Conv1D, UpSampling1D, Flatten, BatchNormalization, Input, Reshape
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

class Plugin:
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3, 
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'l2_reg': 1e-2,     # L2 regularization factor
        'activation': 'tanh'
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
        """
        Configures and builds the decoder model as the exact mirror of the encoder.
        
        Parameters:
            interface_size (int): The latent dimension (must equal the encoder’s output).
            output_shape (int or tuple): The original encoder input shape. If tuple, the first element (window size)
                                         is used and the second element is the original feature count.
            num_channels (int): Number of channels in the original input.
            encoder_output_shape (tuple): The encoder pre-flatten shape (T, F), where T = time steps after pooling and
                                          F is the number of filters from the last Conv1D.
            use_sliding_windows (bool): If True, the decoder output retains temporal dimensions.
        """
        print(f"[DEBUG] Starting decoder configuration with interface_size={interface_size}, output_shape={output_shape}, num_channels={num_channels}, encoder_output_shape={encoder_output_shape}, use_sliding_windows={use_sliding_windows}")
        
        self.params['interface_size'] = interface_size
        # If output_shape is a tuple, extract window size and original feature count.
        if isinstance(output_shape, tuple):
            window_size, orig_features = output_shape
        else:
            window_size = output_shape
            orig_features = num_channels
        self.params['output_shape'] = window_size

        # Ensure encoder_output_shape is a tuple of length 2.
        if isinstance(encoder_output_shape, tuple) and len(encoder_output_shape) == 1:
            encoder_output_shape = (1, encoder_output_shape[0])
        # encoder_output_shape should be (T, F); for our encoder, we expect (32, 32)
        T, F = encoder_output_shape
        print(f"[DEBUG] Using encoder pre-flatten shape: T={T}, F={F}")
        
        # Compute the flattened dimension that the encoder produced (should be T * F, e.g. 32*32 = 1024).
        flat_dim = T * F

        # Recompute the encoder layer sizes exactly as in the encoder.
        enc_layers = []
        current = self.params['initial_layer_size']
        for i in range(self.params['intermediate_layers']):
            enc_layers.append(current)
            current = max(current // self.params['layer_size_divisor'], interface_size)
        enc_layers.append(interface_size)
        print(f"[DEBUG] Encoder layer sizes (from encoder): {enc_layers}")
        # The mirrored conv filters come from the encoder's conv blocks (exclude the final Dense that maps to interface_size)
        # For example, if enc_layers = [128, 64, 32, 32] then the conv blocks used filters 128, 64, 32.
        mirror_filters = enc_layers[:-1][::-1]  # e.g., [32, 64, 128]
        print(f"[DEBUG] Mirrored decoder conv filter sizes: {mirror_filters}")
        
        # Build the decoder model.
        self.model = Sequential(name="decoder_cnn_model")
        # Expand latent vector to flat_dim and reshape to (T, F)
        self.model.add(Dense(units=flat_dim,
                             activation=self.params['activation'],
                             kernel_initializer=GlorotUniform(),
                             kernel_regularizer=l2(self.params['l2_reg']),
                             input_shape=(interface_size,)))
        self.model.add(Reshape((T, F), name="reshape"))
        
        # Now, we need to invert the three pooling blocks.
        # The encoder had three blocks:
        # Block 1: Conv1D(filters=128) + MaxPooling1D(pool_size=2) → output shape (256,128) -> (128,128)
        # Block 2: Conv1D(filters=64) + MaxPooling1D(pool_size=2) → (128,64) -> (64,64)
        # Block 3: Conv1D(filters=32) + MaxPooling1D(pool_size=2) → (64,32) -> (32,32)
        # To mirror, we use UpSampling1D (size=2) followed by a Conv1D layer.
        # We'll use the mirror_filters computed above: e.g. [32, 64, 128].
        for idx, filt in enumerate(mirror_filters):
            # Upsample the time dimension by 2.
            self.model.add(UpSampling1D(size=2))
            # Then apply a Conv1D layer to mirror the Conv1D block.
            self.model.add(Conv1D(
                filters=filt,
                kernel_size=3,
                padding='same',
                activation='relu',
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(self.params['l2_reg']),
                name=f"conv1d_mirror_{idx+1}"
            ))
        # At this point, the time dimension should be upsampled from T (e.g. 32) back to window_size (e.g. 256).
        # Finally, map from the last mirrored filter dimension to the original feature count.
        self.model.add(Conv1D(
            filters=orig_features,
            kernel_size=3,
            padding='same',
            activation='tanh',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(self.params['l2_reg']),
            name="decoder_output_conv1d"
        ))
        
        final_output_shape = self.model.output_shape
        print(f"[DEBUG] Final Output Shape: {final_output_shape}")
        
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

    def train(self, encoded_data, original_data):
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        original_data = original_data.reshape((original_data.shape[0], -1))
        early_stopping = EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1)
        self.model.fit(encoded_data, original_data, epochs=self.params.get('epochs', 100), batch_size=self.params.get('batch_size', 128), verbose=1, callbacks=[early_stopping], validation_split=0.2)

    def decode(self, encoded_data, use_sliding_windows, original_feature_size):
        print(f"[decode] Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data)
        if not use_sliding_windows:
            decoded_data = decoded_data.reshape((decoded_data.shape[0], original_feature_size))
            print(f"[decode] Reshaped decoded data to match original feature size: {decoded_data.shape}")
        else:
            print(f"[decode] Decoded data shape: {decoded_data.shape}")
        return decoded_data

    def save(self, file_path):
        self.model.save(file_path)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Predictor model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], -1))
        reconstructed_data = reconstructed_data.reshape((original_data.shape[0], -1))
        mse = np.mean(np.square(original_data - reconstructed_data))
        return mse

    def calculate_mae(self, original_data, reconstructed_data):
        print(f"Calculating MAE for shapes: original_data={original_data.shape}, reconstructed_data={reconstructed_data.shape}")
        if original_data.shape != reconstructed_data.shape:
            raise ValueError(f"Shape mismatch in calculate_mae: original_data={original_data.shape}, reconstructed_data={reconstructed_data.shape}")
        original_data_flat = original_data.reshape(-1)
        reconstructed_data_flat = reconstructed_data.reshape(-1)
        mae = np.mean(np.abs(original_data_flat - reconstructed_data_flat))
        return mae

    def calculate_r2(self, y_true, y_pred):
        print(f"Calculating R² for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch in calculate_r2: y_true={y_true.shape}, y_pred={y_pred.shape}")
        ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=1, keepdims=True)) ** 2, axis=1)
        r2_scores = 1 - (ss_res / ss_tot)
        r2_scores = np.where(ss_tot == 0, 0, r2_scores)
        r2 = np.mean(r2_scores)
        print(f"Calculated R²: {r2}")
        return r2

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    # For example, assume:
    # - The original encoder input shape was (256, 44).
    # - The encoder produced a pre-flatten shape of (32, 32) (i.e. T=32, F=32).
    # - The number of channels originally was 44.
    plugin.configure_size(interface_size=32, output_shape=(256, 44), num_channels=44, encoder_output_shape=(32, 32), use_sliding_windows=True)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
