"""RNN decoder plugin: upsamples the latent sequence back to the window with recurrent layers."""

import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, SimpleRNN, GRU, UpSampling1D, Conv1D, Cropping1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K


#: the recurrent cells this plugin can stack, selected by the `rnn_type` parameter
RNN_CELLS = {"simple_rnn": SimpleRNN, "gru": GRU}


class Plugin:
    """Decoder mirroring the `rnn` encoder, back to the window.

    Same interface as the other decoder plugins in this package. The latent sequence is doubled in length
    twice, passed through the same kind of recurrent cell the encoder used, projected to `num_channels` by
    a width-1 convolution and cropped to the window length when upsampling overshoots it.
    """

    plugin_params = {
        'intermediate_layers': 3,
        'learning_rate': 0.00002,
        'dropout_rate': 0.001,
        'rnn_type': 'simple_rnn',
        'activation': 'tanh',
    }

    plugin_debug_vars = ['interface_size', 'output_shape', 'intermediate_layers', 'rnn_type']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def _cell(self):
        """The recurrent layer class named by `rnn_type`; unknown names are refused rather than guessed."""
        name = str(self.params.get("rnn_type", "simple_rnn")).lower()
        if name not in RNN_CELLS:
            raise ValueError(f"Unknown rnn_type {name!r}; the declared options are {sorted(RNN_CELLS)}")
        return RNN_CELLS[name]

    def configure_size(self, interface_size, output_shape, num_channels, encoder_output_shape,
                       use_sliding_windows, config=None):
        """
        Configure the decoder from the encoder's output shape.

        Args:
            interface_size (int): Dimension of the bottleneck layer.
            output_shape (int): Length of the reconstructed sequence (the window).
            num_channels (int): Number of channels to reconstruct.
            encoder_output_shape (tuple): `(sequence_length, features)` the encoder produces.
            use_sliding_windows (bool): Whether sliding windows are being used.
            config (dict): Merged run configuration; `window_size` and `activation` are read when present.
        """
        config = dict(config or {})
        print(f"[DEBUG] Starting decoder configuration with interface_size={interface_size}, "
              f"output_shape={output_shape}, num_channels={num_channels}, "
              f"encoder_output_shape={encoder_output_shape}, use_sliding_windows={use_sliding_windows}")

        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        window_size = config.get("window_size", output_shape)
        activation = config.get("activation", self.params.get("activation", "tanh"))
        latent_seq_len, latent_features = encoder_output_shape
        cell = self._cell()

        decoder_input = Input(shape=(latent_seq_len, latent_features), name="decoder_input")

        x = UpSampling1D(size=2, name="decoder_upsampling_1")(decoder_input)
        x = cell(latent_features, activation=activation, return_sequences=True, name="decoder_rnn_1")(x)
        x = UpSampling1D(size=2, name="decoder_upsampling_2")(x)
        x = cell(latent_features, activation=activation, return_sequences=True, name="decoder_rnn_2")(x)

        outputs = Conv1D(filters=num_channels, kernel_size=1, padding='same', activation='linear',
                         name="decoder_output_projection_conv1d")(x)

        produced = K.int_shape(outputs)[1]
        if produced is not None and produced > window_size:
            extra = produced - window_size
            outputs = Cropping1D(cropping=(extra // 2, extra - extra // 2), name="decoder_final_cropping")(outputs)
            print(f"[DEBUG] Decoder cropped {extra} steps to reach window_size={window_size}")
        elif produced is not None and produced < window_size:
            print(f"[WARN] Decoder produces {produced} steps, shorter than window_size={window_size}")

        print(f"[DEBUG] Final Decoder Output shape: {K.int_shape(outputs)}")

        self.model = Model(inputs=decoder_input, outputs=outputs, name="decoder")

        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )

        self.model.compile(
            optimizer=adam_optimizer,
            loss=Huber(),
            metrics=['mse', 'mae'],
            run_eagerly=False
        )
        print(f"[DEBUG] Model compiled successfully.")

    def train(self, encoded_data, original_data, config=None):
        config = dict(config or {})
        early_stopping = EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1)
        self.model.fit(encoded_data, original_data,
                       epochs=self.params.get('epochs', config.get('epochs', 1)),
                       batch_size=config.get("batch_size", 32),
                       verbose=1, callbacks=[early_stopping], validation_split=0.2)

    def decode(self, encoded_data, use_sliding_windows=True, original_feature_size=None):
        print(f"[decode] Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data)
        if not use_sliding_windows and original_feature_size is not None:
            decoded_data = decoded_data.reshape((decoded_data.shape[0], original_feature_size))
        print(f"[decode] Decoded data shape: {decoded_data.shape}")
        return decoded_data

    def save(self, file_path):
        self.model.save(file_path)
        print(f"Decoder model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Decoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], -1))
        reconstructed_data = reconstructed_data.reshape((original_data.shape[0], -1))
        return np.mean(np.square(original_data - reconstructed_data))


# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=4, output_shape=32, num_channels=3,
                          encoder_output_shape=(8, 12), use_sliding_windows=True,
                          config={"window_size": 32})
    print(f"Debug Info: {plugin.get_debug_info()}")
