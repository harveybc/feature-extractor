import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal

from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Bidirectional, LSTM

class Plugin:
    """
    An encoder plugin for a VAE, outputting z_mean and z_log_var.
    """

    plugin_params = {
        "activation": "tanh",
        'learning_rate': 0.00002,
        'dropout_rate': 0.001,
        "initial_layer_size": 48,
        "layer_size_divisor": 2,
        "l2_reg": 5e-5
    }

    plugin_debug_vars = ['input_shape', 'interface_size_config']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None
        self.shape_before_flatten_for_decoder = None 
        self.params['interface_size_config'] = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, input_shape, interface_size, num_channels, use_sliding_windows, config=None):
        """
        Configure the VAE encoder based on input shape, interface size (latent_dim), and channel dimensions.
        
        Args:
            input_shape (int): Length of the sequence or row (window_size).
            interface_size (int): Dimension of the latent space (z_mean, z_log_var will each have this dim).
            num_channels (int): Number of input channels.
            use_sliding_windows (bool): Whether sliding windows are being used.
            config (dict): Configuration dictionary.
        """
        print(f"[DEBUG VAE Encoder] Starting encoder configuration with input_shape(window_size)={input_shape}, interface_size(latent_dim)={interface_size}, num_channels={num_channels}, use_sliding_windows={use_sliding_windows}")

        self.params['input_shape'] = input_shape
        self.params['interface_size_config'] = interface_size

        window_size = config.get("window_size", input_shape)
        merged_units = config.get("initial_layer_size", 64)
        branch_units = merged_units // config.get("layer_size_divisor", 2)
        lstm_units = branch_units // config.get("layer_size_divisor", 2)
        activation = config.get("activation", "tanh")
        l2_reg = config.get("l2_reg", self.params.get("l2_reg", 1e-6))

        inputs = Input(shape=(window_size, num_channels), name="encoder_input_vae")

        merged = Conv1D(
            filters=merged_units, kernel_size=3, strides=2, activation='linear', name="conv_merged_features_1",
            kernel_regularizer=l2(l2_reg)
        )(inputs)
        merged = Conv1D(
            filters=branch_units, kernel_size=3, strides=2, padding='same', activation=activation, name="conv_merged_features_2",
            kernel_regularizer=l2(l2_reg)
        
        )(merged)
        merged = Conv1D(
            filters=lstm_units, kernel_size=3, strides=2, padding='same', activation=activation, name="conv_merged_features_3",
            kernel_regularizer=l2(l2_reg)
        
        )(merged)
        
        conv_output_before_flatten = Conv1D(
            filters=lstm_units//2, kernel_size=3, strides=2, padding='same', activation=activation,
            name="conv_output_before_flatten",
            kernel_regularizer=l2(l2_reg)
        
        )(merged)
        
        self.shape_before_flatten_for_decoder = conv_output_before_flatten.shape[1:]
        print(f"[DEBUG VAE Encoder] Shape before flatten for decoder: {self.shape_before_flatten_for_decoder}")

        flattened_output = Flatten(name="encoder_flatten_vae")(conv_output_before_flatten)
        
        z_mean = Dense(interface_size, name='z_mean')(flattened_output)
        z_log_var = Dense(interface_size, name='z_log_var')(flattened_output)

        print(f"[DEBUG VAE Encoder] z_mean shape: {z_mean.shape}, z_log_var shape: {z_log_var.shape}")

        self.encoder_model = Model(inputs=inputs, outputs=[z_mean, z_log_var], name="vae_encoder")
        
        print(f"[DEBUG VAE Encoder] VAE Encoder model built (outputs z_mean, z_log_var). Not compiled standalone.")

    def train(self, data):
        num_channels = data.shape[-1] if len(data.shape) > 2 else 1
        input_shape = data.shape[1]
        interface_size = self.params.get('interface_size', 4)

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)

        self.configure_size(input_shape, interface_size, num_channels)

        print(f"Training encoder with data shape: {data.shape}")
        early_stopping = EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True)
        self.encoder_model.fit(data, data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1, callbacks=[early_stopping], validation_split = 0.2)
        print("Training completed.")

    def encode(self, data):
        if not self.encoder_model:
            raise ValueError("Encoder model is not configured or loaded.")
        print(f"[VAE Encoder] Encoding data with shape: {data.shape}")
        encoded_outputs = self.encoder_model.predict(data)
        print(f"[VAE Encoder] Produced {len(encoded_outputs)} outputs. z_mean shape: {encoded_outputs[0].shape}, z_log_var shape: {encoded_outputs[1].shape}")
        return encoded_outputs

    def save(self, file_path):
        if self.encoder_model:
            save_model(self.encoder_model, file_path)
            print(f"VAE Encoder model saved to {file_path}")
        else:
            print("VAE Encoder model not available to save.")

    def load(self, file_path):
        self.encoder_model = load_model(file_path, compile=False)
        print(f"VAE Encoder model (outputs z_mean, z_log_var) loaded from {file_path}")
        
        try:
            conv_layer = self.encoder_model.get_layer("conv_output_before_flatten")
            self.shape_before_flatten_for_decoder = conv_layer.output_shape[1:]
            
            z_mean_layer = self.encoder_model.get_layer("z_mean")
            self.params['interface_size_config'] = z_mean_layer.output_shape[-1]
            print(f"[DEBUG VAE Encoder Load] Reconstructed shape_before_flatten: {self.shape_before_flatten_for_decoder}, interface_size: {self.params['interface_size_config']}")
        except ValueError as e:
            print(f"[WARNING VAE Encoder Load] Could not fully reconstruct internal shapes from loaded model. Error: {e}. Ensure layer names are consistent if this is an issue.")

if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(input_shape=128, interface_size=4)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
