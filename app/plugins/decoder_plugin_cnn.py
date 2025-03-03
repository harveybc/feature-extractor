import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Conv1D, UpSampling1D, Flatten, BatchNormalization, Reshape, Concatenate, Input
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
        'l2_reg': 1e-2,
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
        debug_info.update(self.get_debug_info())

    def build_decoder(self, latent_input, skip_tensors, output_shape, encoder_output_shape):
        """
        Builds the decoder model as a mirror of the encoder.
        
        Args:
            latent_input: Keras tensor for the latent vector (shape: (None, interface_size)).
            skip_tensors: List of skip connection tensors from the encoder.
            output_shape (tuple): Original input shape, e.g. (window_size, original_features).
            encoder_output_shape (tuple): Encoder pre-flatten shape (T, F).
        
        Returns:
            Keras tensor for the decoder output.
        """
        # If output_shape is a tuple, extract window_size and original feature count.
        if isinstance(output_shape, tuple):
            window_size, orig_features = output_shape
        else:
            window_size = output_shape
            orig_features = None  # Should not occur for CNN

        T, F = encoder_output_shape  # e.g., (16, 32)
        flat_dim = T * F  # e.g., 16*32 = 512

        # Expand latent vector to flat_dim and reshape to (T, F)
        x = Dense(units=flat_dim,
                  activation=self.params['activation'],
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(self.params['l2_reg']))(latent_input)
        x = Reshape((T, F), name="reshape")(x)

        # Recompute encoder layer sizes as in the encoder.
        enc_layers = []
        current = self.params['initial_layer_size']
        for i in range(self.params['intermediate_layers']):
            enc_layers.append(current)
            current = max(current // self.params['layer_size_divisor'], self.params['interface_size'])
        enc_layers.append(self.params['interface_size'])
        # Mirror conv filters from encoder's conv blocks (exclude the final Dense mapping)
        mirror_filters = enc_layers[:-1][::-1]  # For example, if enc_layers = [128, 64, 32, 32] then mirror_filters = [32, 64, 128]
        
        # For each intermediate layer, upsample, concatenate corresponding skip tensor, and apply Conv1D + BN.
        for idx in range(self.params['intermediate_layers']):
            x = UpSampling1D(size=2, name=f"upsample_{idx+1}")(x)
            # Use corresponding skip tensor from encoder (reverse order)
            if skip_tensors and idx < len(skip_tensors):
                skip = skip_tensors[-(idx+1)]
                x = Concatenate(axis=-1, name=f"skip_concat_{idx+1}")([x, skip])
            filt = mirror_filters[idx] if idx < len(mirror_filters) else mirror_filters[-1]
            x = Conv1D(filters=filt,
                       kernel_size=3,
                       padding='same',
                       activation='relu',
                       kernel_initializer=HeNormal(),
                       kernel_regularizer=l2(self.params['l2_reg']),
                       name=f"conv1d_mirror_{idx+1}")(x)
            x = BatchNormalization(name=f"bn_decoder_{idx+1}")(x)
        # Final mapping: map to original feature count.
        output = Conv1D(filters=orig_features,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='tanh',
                        kernel_initializer=GlorotUniform(),
                        kernel_regularizer=l2(self.params['l2_reg']),
                        name="decoder_output_conv1d")(x)
        return output

    def configure_size(self, interface_size, output_shape, num_channels, encoder_output_shape, use_sliding_windows, encoder_skip_connections):
        """
        Configures and builds the decoder model as the mirror of the encoder using the encoder's skip tensors.
        
        Args:
            interface_size (int): The latent dimension.
            output_shape (int or tuple): Original input shape; if tuple, first element is window size, second is feature count.
            num_channels (int): Number of channels in the original input.
            encoder_output_shape (tuple): Encoder pre-flatten shape (T, F).
            use_sliding_windows (bool): Whether sliding windows are used.
            encoder_skip_connections (list): List of skip connection tensors from the encoder.
        """
        # IMPORTANT: Ensure that 'interface_size' is set in self.params.
        self.params['interface_size'] = interface_size

        if isinstance(output_shape, tuple):
            window_size, orig_features = output_shape
        else:
            window_size = output_shape
            orig_features = num_channels
        self.params['output_shape'] = window_size

        if isinstance(encoder_output_shape, tuple) and len(encoder_output_shape) == 1:
            encoder_output_shape = (1, encoder_output_shape[0])
        T, F = encoder_output_shape
        print(f"[DEBUG] Using encoder pre-flatten shape: T={T}, F={F}")

        print(f"[DEBUG] Starting decoder configuration with interface_size={interface_size}, output_shape={output_shape}, num_channels={num_channels}, encoder_output_shape={encoder_output_shape}, use_sliding_windows={use_sliding_windows}")
        
        # Build decoder using the Functional API.
        latent_input = Input(shape=(interface_size,), name="decoder_latent")
        # Here, we reuse the encoder_skip_connections directly.
        output = self.build_decoder(latent_input, encoder_skip_connections, output_shape, encoder_output_shape)
        self.model = Model(inputs=[latent_input] + encoder_skip_connections, outputs=output, name="decoder_cnn_model")
        print(f"[DEBUG] Final Output Shape: {self.model.output_shape}")

        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-2,
            amsgrad=False
        )
        self.model.compile(optimizer=adam_optimizer,
                           loss=Huber(),
                           metrics=['mse', 'mae'],
                           run_eagerly=False)
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
