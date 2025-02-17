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
        'initial_layer_size': 32,          # Base hidden units in the first LSTM layer
        'layer_size_divisor': 2,
        'l2_reg': 1e-2,
        'learning_rate': 0.0001
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
            input_shape (int or tuple): 
                • If int (and not sliding), it represents the number of features; we then assume a single time step.
                • If tuple, it is (time_steps, num_channels).
            encoding_dim (int): Dimension of the latent space.
            num_channels (int, optional): Number of features per time step (used if input_shape is int).
            use_sliding_windows (bool): If True, input_shape is treated as (time_steps, num_channels).
        """
        # When not using sliding windows and input_shape is an int,
        # we now interpret it as the number of features and set time_steps=1.
        if isinstance(input_shape, int) and not use_sliding_windows:
            if num_channels is None:
                num_channels = input_shape  # Here, the entire int is the feature dimension.
            else:
                # In this branch, we assume input_shape represents feature count.
                num_channels = num_channels
            input_shape = (1, num_channels)
        elif isinstance(input_shape, int) and use_sliding_windows:
            # If sliding windows are used, the caller should provide a tuple.
            raise ValueError("[configure_size] In sliding windows mode, input_shape must be a tuple (time_steps, num_channels).")
        # Otherwise, if input_shape is already a tuple, we leave it as is.
        self.params['input_shape'] = input_shape
        self.params['encoding_dim'] = encoding_dim

        intermediate_layers = self.params.get('intermediate_layers', 3)
        initial_layer_size = self.params.get('initial_layer_size', 32)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        l2_reg = self.params.get('l2_reg', 1e-2)
        learning_rate = self.params.get('learning_rate', 0.0001)

        # Compute the LSTM layer sizes.
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
        # Add LSTM layers with return_sequences=True for all but final layer.
        for idx, size in enumerate(layers[:-1], start=1):
            x = LSTM(units=size, activation='tanh', recurrent_activation='sigmoid',
                     return_sequences=True, name=f"lstm_layer_{idx}")(x)
        # Final LSTM layer without return_sequences.
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

    def train(self, data, validation_data):
        if self.encoder_model is None:
            raise ValueError("[train] Encoder model is not yet configured. Call configure_size first.")
        print(f"[train] Training data shape: {data.shape}, validation shape: {validation_data.shape}")
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.encoder_model.fit(data, data, epochs=self.params.get('epochs',600),
                                          batch_size=self.params.get('batch_size',128),
                                          validation_data=(validation_data, validation_data),
                                          callbacks=[early_stopping], verbose=1)
        print("[train] Training completed.")
        return history

    def encode(self, data):
        if self.encoder_model is None:
            raise ValueError("[encode] Encoder model is not configured.")
        print(f"[encode] Encoding data with shape: {data.shape}")
        encoded_data = self.encoder_model.predict(data, verbose=1)
        print(f"[encode] Encoded data shape: {encoded_data.shape}")
        return encoded_data

    def save(self, file_path):
        if self.encoder_model is None:
            raise ValueError("[save] Encoder model is not configured.")
        save_model(self.encoder_model, file_path)
        print(f"[save] Encoder model saved to {file_path}")

    def load(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"[load] Encoder model loaded from {file_path}")

if __name__ == "__main__":
    plugin = Plugin()
    # For example, if we have 8 features (non-sliding mode), we set input_shape=8 and num_channels=8.
    # The resulting input shape will be (1, 8) so that each sample is a single time step with 8 features.
    plugin.configure_size(input_shape=8, encoding_dim=4, num_channels=8, use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
