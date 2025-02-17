import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Flatten, Reshape, GlobalAveragePooling1D, LayerNormalization, Dropout, Add, TimeDistributed
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber

class Plugin:
    """
    A transformer-based decoder plugin.
    It mirrors the encoder's structure by reversing the layer sizes.
    Batch normalization may be omitted in the decoder to allow full reconstruction.
    """
    plugin_params = {
        'intermediate_layers': 1,
        'layer_size_divisor': 2,
        'ff_dim_divisor': 2,
        'learning_rate': 0.00001,
        'dropout_rate': 0.1,
        'initial_layer_size': 128,
    }
    plugin_debug_vars = ['interface_size', 'output_shape', 'intermediate_layers']

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

    def configure_size(self, interface_size, output_shape):
        """
        Configures the transformer-based decoder.
        Args:
            interface_size (int): Size of the latent vector (decoder input).
            output_shape (int): Dimension of the original input (to be reconstructed).
                For sliding window, this is the flattened dimension.
        """
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        # Compute intermediate sizes similar to the encoder (using initial_layer_size and layer_size_divisor)
        layer_sizes = []
        current_size = self.params.get('initial_layer_size', 128)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        int_layers = self.params.get('intermediate_layers', 1)
        for i in range(int_layers):
            layer_sizes.append(current_size)
            current_size = max(current_size // layer_size_divisor, interface_size)
        layer_sizes.append(interface_size)
        # For the decoder, we reverse the order (mirror)
        layer_sizes.reverse()
        print(f"[configure_size] Transformer decoder layer sizes (mirrored): {layer_sizes}")
        
        # Input layer: the decoder takes the latent vector
        inputs = Input(shape=(interface_size,), name="decoder_input")
        # Reshape latent vector to a sequence (for transformer processing)
        x = Reshape((interface_size, 1))(inputs)

        # Apply transformer blocks for each layer size in the mirrored list.
        for size in layer_sizes:
            ff_dim = max(size // self.params.get('ff_dim_divisor', 2), 1)
            dropout_rate = self.params.get('dropout_rate', 0.1)
            # Projection dense layer
            x = Dense(size, name="proj_dense")(x)
            # Multi-head attention block
            # Heuristic for number of heads:
            if size < 64:
                num_heads = 2
            elif size < 128:
                num_heads = 4
            else:
                num_heads = 8
            x = MultiHeadAttention(head_num=num_heads, name="multi_head")(x)
            x = LayerNormalization(epsilon=1e-6, name="layer_norm_1")(x)
            x = Dropout(dropout_rate, name="dropout_1")(x)
            # Feed-forward network
            ffn_output = Dense(ff_dim, activation='relu', kernel_initializer=HeNormal(), name="ffn_dense_1")(x)
            ffn_output = Dense(size, name="ffn_dense_2")(ffn_output)
            ffn_output = Dropout(dropout_rate, name="dropout_2")(ffn_output)
            x = Add(name="residual_add")([x, ffn_output])
            x = LayerNormalization(epsilon=1e-6, name="layer_norm_2")(x)

        x = Flatten()(x)
        outputs = Dense(output_shape, activation='tanh', kernel_initializer=GlorotUniform(), name="decoder_output")(x)
        self.model = Model(inputs=inputs, outputs=outputs, name="decoder_transformer")
        adam_optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.00001), beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
        print("[configure_size] Transformer Decoder Model Summary:")
        self.model.summary()

    def train(self, encoded_data, original_data):
        print(f"Training transformer decoder with encoded data shape: {encoded_data.shape} and original data shape: {original_data.shape}")
        self.model.fit(encoded_data, original_data, epochs=self.params.get('epochs',200), batch_size=self.params.get('batch_size',128), verbose=1)
        print("Training completed.")

    def decode(self, encoded_data):
        print(f"Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data)
        print(f"Decoded data shape: {decoded_data.shape}")
        return decoded_data

    def save(self, file_path):
        save_model(self.model, file_path)
        print(f"Decoder model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Decoder model loaded from {file_path}")

if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=4, output_shape=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
