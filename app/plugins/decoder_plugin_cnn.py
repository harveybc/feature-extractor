import numpy as np
from keras.models import Sequential, load_model, save_model, Model
from keras.layers import Dense, Conv1D, Conv1DTranspose, Reshape, Flatten, BatchNormalization, MaxPooling1D, Input
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

class Plugin:
    # Use the exact same plugin parameters as the encoder.
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
            interface_size (int): Desired latent dimension (must equal the encoder’s output).
            output_shape (int): The original window size.
            num_channels (int): Number of channels in the original input.
            encoder_output_shape (tuple): Pre-flatten shape from the encoder (e.g. (T, F)).
            use_sliding_windows (bool): If True, the output retains temporal dimensions.
        """
        print(f"[DEBUG] Starting decoder configuration with interface_size={interface_size}, output_shape={output_shape}, num_channels={num_channels}, encoder_output_shape={encoder_output_shape}, use_sliding_windows={use_sliding_windows}")
        
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        # Ensure encoder_output_shape is a tuple of length 2.
        if isinstance(encoder_output_shape, tuple) and len(encoder_output_shape) == 1:
            encoder_output_shape = (1, encoder_output_shape[0])
        
        # Extract sequence_length and num_filters from encoder_output_shape.
        sequence_length, num_filters = encoder_output_shape
        print(f"[DEBUG] Extracted sequence_length={sequence_length}, num_filters={num_filters} from encoder_output_shape.")

        num_intermediate_layers = self.params['intermediate_layers']
        print(f"[DEBUG] Number of intermediate layers={num_intermediate_layers}")
        
        # Use the l2 regularization factor from plugin parameters.
        l2_reg = self.params['l2_reg']
        layers = [output_shape * 2]
        current_size = output_shape * 2
        for i in range(num_intermediate_layers - 1):
            next_size = current_size // self.params['layer_size_divisor']
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
            activation='tanh',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            padding='same',
            input_shape=(sequence_length, num_filters)
        ))

        for idx, size in enumerate(layer_sizes[1:], start=1):
            strides = 2 if idx < len(layer_sizes) - 1 else 1
            self.model.add(Conv1DTranspose(
                filters=size,
                kernel_size=3,
                strides=strides,
                padding='same',
                activation='tanh',
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg)
            ))

        if use_sliding_windows:
            # For sliding windows, map to original channel dimension.
            self.model.add(Conv1DTranspose(
                filters=num_channels,
                kernel_size=3,
                padding='same',
                activation='tanh',
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg),
                name="decoder_output"
            ))
        else:
            # For row-by-row data, flatten and use a Dense layer.
            self.model.add(Flatten(name="decoder_flatten"))
            self.model.add(Dense(
                units=output_shape,
                activation='linear',
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg),
                name="decoder_dense_output"
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
    plugin.configure_size(interface_size=4, output_shape=128, num_channels=8, encoder_output_shape=(32, 32), use_sliding_windows=True)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
