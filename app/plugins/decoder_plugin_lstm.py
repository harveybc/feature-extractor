import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Dense, Input, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

class Plugin:
    """
    An LSTM-based encoder plugin.
    Configurable similarly to the ANN plugin.
    """

    plugin_params = {
        'intermediate_layers': 3,        # Number of LSTM layers before the final projection
        'initial_layer_size': 32,        # Base hidden units in first LSTM layer
        'layer_size_divisor': 2,
        'l2_reg': 1e-2
    }

    plugin_debug_vars = ['input_shape', 'encoding_dim']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var, None) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def configure_size(self, input_shape, encoding_dim, num_channels=None, use_sliding_windows=False):
        """
        Configures the LSTM-based encoder.
        Args:
            input_shape (int or tuple): If int, represents time_steps; if tuple, (time_steps, num_channels).
            encoding_dim (int): Dimension of the latent space.
            num_channels (int, optional): Number of features. If input_shape is an int and sliding windows
                are not used, this will be used to set the number of features.
            use_sliding_windows (bool, optional): If True, input_shape is treated as (time_steps, num_channels).
        """
        # Modified: if input_shape is an int and we're NOT using sliding windows,
        # use the provided num_channels (or default to 1 if not provided)
        if isinstance(input_shape, int):
            if use_sliding_windows:
                input_shape = (input_shape, num_channels if num_channels else 1)
            else:
                input_shape = (input_shape, num_channels if num_channels else 1)
        self.params['input_shape'] = input_shape
        self.params['encoding_dim'] = encoding_dim

        intermediate_layers = self.params.get('intermediate_layers', 3)
        initial_layer_size = self.params.get('initial_layer_size', 32)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        l2_reg = self.params.get('l2_reg', 1e-2)
        learning_rate = self.params.get('learning_rate', 0.0001)

        layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        layers.append(encoding_dim)
        print(f"[configure_size] LSTM Layer sizes: {layers}")
        print(f"[configure_size] LSTM input shape: {input_shape}")

        encoder_input = Input(shape=input_shape, name="encoder_input")
        x = encoder_input

        # Add LSTM layers with return_sequences=True for all but final
        for idx, size in enumerate(layers[:-1], start=1):
            x = LSTM(units=size, activation='tanh', recurrent_activation='sigmoid',
                     return_sequences=True, name=f"lstm_layer_{idx}")(x)
        # Final LSTM layer (without return_sequences)
        if len(layers) >= 2:
            x = LSTM(units=layers[-2], activation='tanh', recurrent_activation='sigmoid',
                     return_sequences=False, name="lstm_layer_final")(x)
        x = BatchNormalization(name="batch_norm_final")(x)
        encoder_output = Dense(units=layers[-1], activation='linear',
                               kernel_initializer=GlorotUniform(),
                               kernel_regularizer=l2(l2_reg),
                               name="encoder_output")(x)

        self.encoder_model = Model(inputs=encoder_input, outputs=encoder_output, name="encoder_lstm")
        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.encoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Encoder Model Summary:")
        self.encoder_model.summary()

if __name__ == "__main__":
    # Example: For an 8-feature time series with 128 time steps, using non-sliding windows.
    # We pass input_shape=128 and num_channels=8.
    plugin = Plugin()
    plugin.configure_size(input_shape=128, encoding_dim=4, num_channels=8, use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
