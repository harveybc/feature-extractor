"""RNN encoder plugin: a stack of simple recurrent layers over the input window."""

import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Input, SimpleRNN, GRU, AveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber


#: the recurrent cells this plugin can stack, selected by the `rnn_type` parameter
RNN_CELLS = {"simple_rnn": SimpleRNN, "gru": GRU}


class Plugin:
    """Encoder of two recurrent layers (SimpleRNN or GRU).

    Same interface as the other encoder plugins in this package: `configure_size` builds and compiles the
    Keras model, `encode` runs it over a batch of windows, `save`/`load` persist it. The window is reduced
    by a factor of two after each recurrent layer, so a window of length W leaves ceil(ceil(W/2)/2) steps of
    `rnn_units` features -- the same reduction the `lstm` encoder applies, so its decoder shape arithmetic
    holds for the matching `rnn` decoder too.
    """

    plugin_params = {
        "activation": "tanh",
        "intermediate_layers": 2,
        "learning_rate": 0.00002,
        "dropout_rate": 0.001,
        "initial_layer_size": 48,
        "layer_size_divisor": 2,
        "l2_reg": 5e-5,
        "rnn_type": "simple_rnn",
    }

    plugin_debug_vars = ['input_shape', 'intermediate_layers', 'rnn_type']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

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

    def configure_size(self, input_shape, interface_size, num_channels, use_sliding_windows, config=None):
        """
        Configure the encoder based on input shape, interface size, and channel dimensions.

        Args:
            input_shape (int): Length of the sequence or row.
            interface_size (int): Dimension of the bottleneck layer.
            num_channels (int): Number of input channels.
            use_sliding_windows (bool): Whether sliding windows are being used.
            config (dict): Merged run configuration; `window_size`, `initial_layer_size`,
                `layer_size_divisor` and `activation` are read from it when present.
        """
        config = dict(config or {})
        print(f"[DEBUG] Starting encoder configuration with input_shape={input_shape}, "
              f"interface_size={interface_size}, num_channels={num_channels}, "
              f"use_sliding_windows={use_sliding_windows}")

        self.params['input_shape'] = input_shape

        window_size = config.get("window_size", input_shape)
        merged_units = config.get("initial_layer_size", self.params.get("initial_layer_size", 48))
        divisor = config.get("layer_size_divisor", self.params.get("layer_size_divisor", 2)) or 1
        branch_units = max(merged_units // divisor, 1)
        rnn_units = max(branch_units // divisor, interface_size, 1)
        activation = config.get("activation", self.params.get("activation", "tanh"))
        cell = self._cell()
        print(f"[DEBUG] Encoder recurrent units: {rnn_units} ({self.params.get('rnn_type')}), "
              f"window_size={window_size}")

        inputs = Input(shape=(window_size, num_channels), name="input_layer")

        x = cell(rnn_units, activation=activation, return_sequences=True, name="feature_rnn_1")(inputs)
        x = AveragePooling1D(pool_size=3, strides=2, padding='same', name="average_pooling_1")(x)
        x = cell(rnn_units, activation=activation, return_sequences=True, name="feature_rnn_2")(x)
        outputs = AveragePooling1D(pool_size=3, strides=2, padding='same', name="average_pooling_2")(x)

        print(f"[DEBUG] Final Output shape: {outputs.shape}")

        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder")

        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )

        self.encoder_model.compile(
            optimizer=adam_optimizer,
            loss=Huber(),
            metrics=['mse', 'mae'],
            run_eagerly=False
        )
        print(f"[DEBUG] Encoder model compiled successfully.")

    def train(self, data, config=None):
        config = dict(config or {})
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        num_channels = data.shape[-1]
        input_shape = data.shape[1]
        interface_size = self.params.get('interface_size', 4)

        if self.encoder_model is None:
            config.setdefault("window_size", input_shape)
            self.configure_size(input_shape, interface_size, num_channels, True, config=config)

        print(f"Training encoder with data shape: {data.shape}")
        early_stopping = EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True)
        self.encoder_model.fit(data, data,
                               epochs=self.params.get('epochs', config.get('epochs', 1)),
                               batch_size=self.params.get('batch_size', config.get('batch_size', 32)),
                               verbose=1, callbacks=[early_stopping], validation_split=0.2)
        print("Training completed.")

    def encode(self, data):
        print(f"Encoding data with shape: {data.shape}")
        encoded_data = self.encoder_model.predict(data)
        print(f"Encoded data shape: {encoded_data.shape}")
        return encoded_data

    def save(self, file_path):
        save_model(self.encoder_model, file_path)
        print(f"Encoder model saved to {file_path}")

    def load(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"Encoder model loaded from {file_path}")


# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(input_shape=32, interface_size=4, num_channels=3,
                          use_sliding_windows=True, config={"window_size": 32})
    print(f"Debug Info: {plugin.get_debug_info()}")
