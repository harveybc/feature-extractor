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

    def configure_size(self, interface_size, input_shape, num_channels, encoder_output_shape, use_sliding_windows):
        # Ensure use_sliding_windows is a proper Boolean.
        use_sliding_windows = str(use_sliding_windows).lower() == 'true'
        self.params['interface_size'] = interface_size
        self.params['input_shape'] = input_shape

        # Retrieve architecture parameters
        intermediate_layers = self.params.get('intermediate_layers', 3)
        initial_layer_size = self.params.get('initial_layer_size', 128)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        l2_reg = self.params.get('l2_reg', 1e-5)
        learning_rate = self.params.get('learning_rate', 0.00002)

        # Recompute encoder's layer sizes using the same logic as in the encoder.
        encoder_layers = []
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            encoder_layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        encoder_layers.append(interface_size)

        # For the decoder, mirror the encoder's intermediate layers (excluding the latent layer).
        decoder_intermediate_layers = encoder_layers[:-1][::-1]
        print(f"[configure_size] Encoder layer sizes: {encoder_layers}")
        print(f"[configure_size] Decoder intermediate layer sizes: {decoder_intermediate_layers}")
        print(f"[configure_size] Original input shape: {input_shape}")

        # Determine final output units: if sliding windows are used and input_shape is a tuple,
        # the final output units equals the product of its dimensions; otherwise, it's input_shape.
        if use_sliding_windows and isinstance(input_shape, tuple):
            final_output_units = int(np.prod(input_shape))
        else:
            final_output_units = input_shape

        # Extract sequence length and number of filters from encoder_output_shape.
        sequence_length, num_filters = encoder_output_shape
        print(f"[configure_size] Extracted sequence_length={sequence_length}, num_filters={num_filters}")

        # Build the decoder model using Conv1DTranspose layers.
        from keras.models import Sequential
        self.model = Sequential(name="decoder_cnn")
        # First Conv1DTranspose layer: input shape must match the encoder's output shape.
        self.model.add(Conv1DTranspose(
            filters=decoder_intermediate_layers[0],
            kernel_size=3,
            strides=1,
            activation='tanh',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            padding='same',
            input_shape=(sequence_length, num_filters)
        ))
        # Add subsequent Conv1DTranspose layers
        layer_sizes = decoder_intermediate_layers[1:]
        for idx, size in enumerate(layer_sizes, start=1):
            # Use stride 2 for all but the final layer
            strides = 2 if idx < len(layer_sizes) else 1
            self.model.add(Conv1DTranspose(
                filters=size,
                kernel_size=3,
                strides=strides,
                padding='same',
                activation='tanh',
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg)
            ))
        # Final projection: if sliding windows are used, output shape equals final_output_units; otherwise, flatten and use Dense.
        if use_sliding_windows:
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
            from keras.layers import Flatten
            self.model.add(Flatten(name="decoder_flatten"))
            self.model.add(Dense(
                units=final_output_units,
                activation='linear',
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg),
                name="decoder_dense_output"
            ))
        final_output_shape = self.model.output_shape
        print(f"[configure_size] Final Output Shape: {final_output_shape}")

        adam_optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )
        self.model.compile(
            optimizer=adam_optimizer,
            loss='mean_squared_error'
        )
        print("[configure_size] Decoder Model Summary:")
        self.model.summary()



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
