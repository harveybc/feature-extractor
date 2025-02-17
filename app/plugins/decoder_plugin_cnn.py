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
    """
    Configures the CNN decoder using Conv1DTranspose layers.
    
    Args:
        interface_size (int): Latent space dimension (decoder input size).
        input_shape (int or tuple): Original input shape fed to the encoder.
            If sliding windows are used and input_shape is a tuple, the final number of units equals the product of dimensions.
            Otherwise (non-sliding), input_shape is the original feature count.
        num_channels (int): Number of input features/channels.
        encoder_output_shape: Output shape of the encoder (excluding batch size); ideally a tuple (sequence_length, num_filters).
        use_sliding_windows (bool): Whether sliding windows are used.
    """
    self.params['interface_size'] = interface_size
    self.params['input_shape'] = input_shape

    # Retrieve architecture parameters.
    intermediate_layers = self.params.get('intermediate_layers', 3)
    initial_layer_size = self.params.get('initial_layer_size', 128)
    layer_size_divisor = self.params.get('layer_size_divisor', 2)
    l2_reg = self.params.get('l2_reg', 1e-5)
    learning_rate = self.params.get('learning_rate', 0.00002)

    # Compute the encoder's layer sizes as in the encoder.
    encoder_layers = []
    current_size = initial_layer_size
    for i in range(intermediate_layers):
        encoder_layers.append(current_size)
        current_size = max(current_size // layer_size_divisor, 1)
    encoder_layers.append(interface_size)
    
    # Mirror the encoder: reverse the intermediate layers (exclude the latent interface itself)
    decoder_intermediate_layers = encoder_layers[:-1][::-1]
    print(f"[configure_size] Encoder layer sizes: {encoder_layers}")
    print(f"[configure_size] Decoder intermediate layer sizes: {decoder_intermediate_layers}")
    print(f"[configure_size] Original input shape: {input_shape}")

    # Determine final output units.
    if use_sliding_windows and isinstance(input_shape, tuple):
        final_output_units = int(np.prod(input_shape))
    else:
        # For non-sliding windows: we assume that the CNN encoder received data of shape (input_shape, input_shape)
        # via tiling. Thus, final output units should be input_shape * input_shape.
        final_output_units = input_shape * num_channels

    # Unpack encoder_output_shape.
    # If it is not a 2-tuple, assume sequence_length = 1.
    if isinstance(encoder_output_shape, tuple) and len(encoder_output_shape) == 2:
        sequence_length, num_filters = encoder_output_shape
    else:
        if isinstance(encoder_output_shape, tuple):
            num_filters = encoder_output_shape[0]
        else:
            num_filters = encoder_output_shape
        sequence_length = 1
    print(f"[configure_size] Using sequence_length={sequence_length}, num_filters={num_filters}")

    # -------------------------------
    # Build the decoder model using Conv1DTranspose layers.
    # -------------------------------
    from keras.models import Model
    from keras.layers import Dense, Reshape, Conv1DTranspose, BatchNormalization, Input

    # Decoder input: latent vector of shape (interface_size,)
    decoder_input = Input(shape=(interface_size,), name="decoder_input")
    x = decoder_input

    # First, map the latent vector to a small “feature map.”
    # For non-sliding windows, we want to recover a 2D structure of shape (input_shape, num_channels).
    # We set the latent feature map size to (time_steps_latent, num_filters).
    # If encoder_output_shape lacks temporal dimension, we use time_steps_latent = 1.
    time_steps_latent = sequence_length  
    x = Dense(time_steps_latent * num_filters, activation='linear',
              kernel_initializer=GlorotUniform(),
              kernel_regularizer=l2(l2_reg),
              name="decoder_dense_to_map")(x)
    x = BatchNormalization(name="decoder_dense_bn")(x)
    x = Reshape((time_steps_latent, num_filters), name="decoder_reshape")(x)
    print(f"[configure_size] After Dense mapping and reshape, feature map shape: {x.shape}")

    # Now add a series of Conv1DTranspose layers to upscale the feature map.
        for i, filters in enumerate(decoder_intermediate_layers, start=1):
            # We use a kernel_size of 3 and a stride of 2 to mirror the downsampling in the encoder.
            x = Conv1DTranspose(filters=filters, kernel_size=3, strides=2, padding='same',
                                activation=LeakyReLU(alpha=0.1),
                                kernel_initializer=HeNormal(),
                                kernel_regularizer=l2(l2_reg),
                                name=f"decoder_conv1dtrans_{i}")(x)
            x = BatchNormalization(name=f"decoder_conv1dtrans_bn_{i}")(x)
            print(f"[configure_size] After Conv1DTranspose layer {i}, shape: {x.shape}")

        # Final layer: set the number of channels to match the original input.
        if use_sliding_windows:
            # For sliding windows, we keep the temporal dimension.
            final_filters = num_channels
            x = Conv1DTranspose(filters=final_filters, kernel_size=3, strides=1, padding='same',
                                activation='linear',
                                kernel_initializer=GlorotUniform(),
                                kernel_regularizer=l2(l2_reg),
                                name="decoder_conv1dtrans_final")(x)
        else:
            # For non-sliding windows, we want a flat output matching final_output_units.
            x = Dense(final_output_units, activation='linear',
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg),
                    name="decoder_dense_output")(x)
            # Then reshape to recover the original 2D structure.
            x = Reshape((input_shape, num_channels), name="decoder_final_reshape")(x)
            print(f"[configure_size] Reshaped decoder output to: ({input_shape}, {num_channels})")

        outputs = x

        self.model = Model(inputs=decoder_input, outputs=outputs, name="decoder_cnn")
        from keras.optimizers import Adam
        adam_optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
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
