import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

class Plugin:
    """
    A decoder plugin that mirrors the LSTM encoder's architecture.
    It expands the latent vector, repeats it to form a sequence, then applies LSTM layers in reversed order
    followed by a TimeDistributed Dense layer.
    """
    plugin_params = {
        'dropout_rate': 0.0,
        'l2_reg': 1e-5,
        'initial_layer_size': 32,
        'intermediate_layers': 3,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001
    }
    plugin_debug_vars = []

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var, None) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def configure_size(self, interface_size, output_time_steps, num_channels=None, encoder_output_shape=None, use_sliding_windows=False):
        """
        Configures the LSTM-based decoder.
        Args:
            interface_size (int): Size of the latent vector (should match the encoder output dimension).
            output_time_steps (int): The desired sequence length of the output.
            num_channels (int, optional): Number of features per time step.
            encoder_output_shape: Not used in this mirror decoder.
            use_sliding_windows (bool): Ignored in this decoder.
        """
        # Here, output_time_steps is an integer representing the desired sequence length.
        self.params['output_shape'] = output_time_steps
        self.params['interface_size'] = interface_size
        if num_channels is None:
            num_channels = 1

        # Compute the encoder layer sizes as used in the encoder.
        initial_layer_size = self.params.get('initial_layer_size', 32)
        intermediate_layers = self.params.get('intermediate_layers', 3)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        encoder_layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            encoder_layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        encoder_layers.append(interface_size)
        print(f"[configure_size] Encoder layer sizes (from encoder): {encoder_layers}")
        # Mirror the encoder LSTM sizes (exclude the latent vector itself).
        decoder_lstm_sizes = list(reversed(encoder_layers[:-1]))
        print(f"[configure_size] Decoder LSTM sizes (mirrored): {decoder_lstm_sizes}")

        self.model = Sequential(name="decoder_lstm")
        # Expand the latent vector via Dense to match the last LSTM unit of the encoder.
        latent_dense_units = encoder_layers[-2]
        self.model.add(Dense(
            latent_dense_units,
            input_shape=(interface_size,),
            activation='tanh',
            kernel_initializer=HeNormal(),
            kernel_regularizer=l2(self.params.get('l2_reg', 1e-2)),
            name="decoder_dense_expand"
        ))
        # Optional dropout after expansion.
        dropout_rate = self.params.get('dropout_rate', 0.1)
        if dropout_rate > 0.0:
            from keras.layers import Dropout
            self.model.add(Dropout(dropout_rate, name="decoder_dropout_after_dense"))
        # Repeat the vector to form a sequence of length equal to output_time_steps.
        from keras.layers import RepeatVector
        self.model.add(RepeatVector(output_time_steps))
        print(f"[configure_size] Added RepeatVector layer with output length: {output_time_steps}")
        # Add mirrored LSTM layers.
        for idx, units in enumerate(decoder_lstm_sizes, start=1):
            from keras.layers import LSTM
            self.model.add(LSTM(
                units=units,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=True,
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(self.params.get('l2_reg', 1e-2)),
                name=f"decoder_lstm_{idx}"
            ))
            if dropout_rate > 0.0:
                from keras.layers import Dropout
                self.model.add(Dropout(dropout_rate, name=f"decoder_dropout_after_lstm_{idx}"))
        # Final TimeDistributed Dense layer to produce output features.
        from keras.layers import TimeDistributed
        self.model.add(TimeDistributed(
            Dense(num_channels,
                  activation='linear',
                  kernel_initializer=GlorotUniform(),
                  kernel_regularizer=l2(self.params.get('l2_reg', 1e-2))),
            name="decoder_output"
        ))
        self.model.compile(
            optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.001),
                           beta_1=0.9, beta_2=0.999, epsilon=1e-7),
            loss='mean_squared_error'
        )
        print("[configure_size] Decoder Model Summary:")
        self.model.summary()

    def train(self, encoded_data, original_data):
        # Ensure original_data has three dimensions (time_steps, features)
        if len(original_data.shape) == 2:
            original_data = np.expand_dims(original_data, axis=-1)
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
        self.model.fit(encoded_data, original_data,
                       epochs=self.params.get('epochs',200),
                       batch_size=self.params.get('batch_size',128),
                       verbose=1,
                       validation_split=0.2,
                       callbacks=[early_stopping])
        print("[train] Training completed.")

    def decode(self, encoded_data):
        print(f"[decode] Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data)
        print(f"[decode] Decoded data shape: {decoded_data.shape}")
        return decoded_data

    def save(self, file_path):
        self.model.save(file_path)
        print(f"[save] Decoder model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"[load] Decoder model loaded from {file_path}")

if __name__ == "__main__":
    plugin = Plugin()
    # For non-sliding LSTM decoder, we assume output_time_steps equals the original time steps (e.g. 8)
    plugin.configure_size(interface_size=4, output_time_steps=8, num_channels=8, encoder_output_shape=(4,), use_sliding_windows=False)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
