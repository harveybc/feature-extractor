import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1DTranspose, GlobalAveragePooling1D, Dense, Input, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber

class Plugin:
    """
    A decoder plugin using a convolutional neural network (CNN) based on Keras.
    This decoder mirrors the encoder's architecture: the intermediate filter sizes are 
    computed using the same parameters as in the encoder (initial_layer_size, intermediate_layers,
    and layer_size_divisor) but in reverse order. Batch normalization is omitted in the decoder
    for improved reconstruction flexibility.
    """
    plugin_params = {
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'l2_reg': 1e-5,
        'learning_rate': 0.00002,
    }
    plugin_debug_vars = ['input_shape', 'interface_size']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value 

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def configure_size(self, input_dim, encoding_dim, num_channels=None, use_sliding_windows=False):
        """
        Configures the CNN-based encoder.
        Args:
            input_dim (int): Length of the input sequence.
            encoding_dim (int): Dimension of the latent space.
            num_channels (int, optional): Number of input channels (if not provided, defaults to 1).
            use_sliding_windows (bool): If True, the input shape is assumed to be (input_dim, num_channels).
        """
        self.params['input_dim'] = input_dim
        self.params['encoding_dim'] = encoding_dim
        if num_channels is None:
            num_channels = 1
        self.params['num_channels'] = num_channels

        # Compute layer sizes using the same method as the ANN plugin:
        intermediate_layers = self.params.get('intermediate_layers', 3)
        initial_layer_size = self.params.get('initial_layer_size', 128)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        l2_reg = self.params.get('l2_reg', 1e-5)
        learning_rate = self.params.get('learning_rate', 0.00002)

        layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        layers.append(encoding_dim)
        print(f"[configure_size] CNN Layer sizes (filters): {layers}")
        print(f"[configure_size] Input sequence length: {input_dim}, Channels: {num_channels}")

        # Define input shape
        cnn_input_shape = (input_dim, num_channels)
        inputs = Input(shape=cnn_input_shape, name="encoder_input")
        x = inputs

        # First Conv1D layer (stride=1)
        x = Conv1D(filters=layers[0], kernel_size=3, strides=1, padding='same',
                activation=LeakyReLU(alpha=0.1),
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(l2_reg),
                name="conv1d_layer_1")(x)
        x = BatchNormalization(name="batch_norm_1")(x)

        # Add intermediate layers with stride=2 for downsampling
        for i, filters in enumerate(layers[1:-1], start=2):
            x = Conv1D(filters=filters, kernel_size=3, strides=2, padding='same',
                    activation=LeakyReLU(alpha=0.1),
                    kernel_initializer=HeNormal(),
                    kernel_regularizer=l2(l2_reg),
                    name=f"conv1d_layer_{i}")(x)
            x = BatchNormalization(name=f"batch_norm_{i}")(x)

        # Final Conv1D layer to produce latent representation
        x = Conv1D(filters=layers[-1], kernel_size=1, strides=1, padding='same',
                activation='linear',
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg),
                name="conv1d_final")(x)
        x = BatchNormalization(name="batch_norm_final")(x)
        # NOTE: Removed GlobalAveragePooling1D so that the output remains 2D (sequence_length, filters)
        outputs = x

        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder_cnn")
        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Encoder Model Summary:")
        self.encoder_model.summary()



    def train(self, data, validation_data):
        if self.model is None:
            raise ValueError("[train] Decoder model is not configured. Call configure_size first.")
        print(f"[train] Starting training with data shape={data.shape}, validation shape={validation_data.shape}")
        history = self.model.fit(data, data, epochs=self.params.get('epochs',200),
                                  batch_size=self.params.get('batch_size',128),
                                  validation_data=(validation_data, validation_data),
                                  verbose=1)
        print("[train] Training completed.")
        return history

    def decode(self, encoded_data):
        if self.model is None:
            raise ValueError("[decode] Decoder model is not configured.")
        print(f"[decode] Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data, verbose=1)
        print(f"[decode] Decoded output shape: {decoded_data.shape}")
        return decoded_data

    def save(self, file_path):
        if self.model is None:
            raise ValueError("[save] Decoder model is not configured.")
        save_model(self.model, file_path)
        print(f"[save] Decoder model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"[load] Decoder model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    # For example, assume input_shape=128 (if sliding windows, it could be (window_size, num_channels))
    plugin.configure_size(interface_size=4, input_shape=128, num_channels=1, encoder_output_shape=(4,), use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
