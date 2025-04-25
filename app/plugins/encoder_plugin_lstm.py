import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input ,Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal

from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.losses import Huber
from keras.layers import Add, LayerNormalization, AveragePooling1D, Bidirectional, LSTM
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K
from keras.layers import Input
import tensorflow as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # Apply sin to even indices; cos to odd indices
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]  # Shape: (1, position, d_model)
    return tf.cast(pos_encoding, dtype=tf.float32)


class Plugin:
    """
    An encoder plugin using a convolutional neural network (CNN) based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        "activation": "tanh",
        'intermediate_layers': 3, 
        'learning_rate': 0.00002,
        'dropout_rate': 0.001,
        "intermediate_layers": 2,
        "initial_layer_size": 48,
        "layer_size_divisor": 2,
        "l2_reg": 5e-5
    }

    plugin_debug_vars = ['input_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.encoder_model = None

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
        Configure the encoder based on input shape, interface size, and channel dimensions.
        
        Args:
            input_shape (int): Length of the sequence or row.
            interface_size (int): Dimension of the bottleneck layer.
            num_channels (int): Number of input channels.
            use_sliding_windows (bool): Whether sliding windows are being used.
        """
        print(f"[DEBUG] Starting encoder configuration with input_shape={input_shape}, interface_size={interface_size}, num_channels={num_channels}, use_sliding_windows={use_sliding_windows}")

        self.params['input_shape'] = input_shape

        # Determine the input shape for the first Conv1D layer
        adjusted_channels = num_channels 

        # Initialize layers array with input_shape
        layers = [adjusted_channels]  # First layer matches the number of input channels
        num_intermediate_layers = self.params['intermediate_layers']

        # Calculate sizes of intermediate layers based on downscaling by 2
        current_size = adjusted_channels
        for i in range(num_intermediate_layers - 1):
            next_size = current_size // 2  # Scale down by half
            if next_size < interface_size:
                next_size = interface_size  # Ensure we don't go below the interface_size
            layers.append(next_size)
            current_size = next_size

        # Append the final layer which is the interface size
        layers.append(interface_size)
        print(f"[DEBUG] Encoder Layer sizes: {layers}")
        
        window_size = config.get("window_size", 288)
        merged_units = config.get("initial_layer_size", 128)
        branch_units = merged_units//config.get("layer_size_divisor", 2)
        lstm_units = branch_units//config.get("layer_size_divisor", 2)
        activation = config.get("activation", "tanh")
        l2_reg = config.get("l2_reg", self.params.get("l2_reg", 1e-6))

        # --- Input Layer ---
        inputs = Input(shape=(window_size, num_channels), name="input_layer")
        x = inputs
        
        # Add positional encoding to capture temporal order
        # get static shape tuple via Keras backend
        last_layer_shape = K.int_shape(x)
        feature_dim = last_layer_shape[-1]
        # get the sequence length from the last layer shape
        seq_length = last_layer_shape[1]
        pos_enc = positional_encoding(seq_length, feature_dim)
        x = x + pos_enc

        
        # --- Self-Attention Block 1 ---
        num_attention_heads = 2
        # get the last layer shape from the merged tensor
        last_layer_shape = K.int_shape(x)
        # get the feature dimension from the last layer shape as the last component of the shape tuple
        feature_dim = last_layer_shape[-1]
        # define key dimension for attention    
        attention_key_dim = feature_dim//num_attention_heads
        # Apply MultiHeadAttention
        attention_output = MultiHeadAttention(
            num_heads=num_attention_heads, # Assumed to be defined
            key_dim=attention_key_dim,     # Assumed to be defined
            kernel_regularizer=l2(l2_reg),
            name=f"multihead_attention_1"
        )(query=x, value=x, key=x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        #AveragePooling1D
        x = AveragePooling1D(pool_size=3, strides=2, padding='same', name=f"average_pooling_1")(x)

        # --- End Self-Attention Block ---
        x = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg),
                    name=f"feature_lstm_1"))(x)

        # --- End Self-Attention Block ---
        x = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg),
                    name=f"feature_lstm_2"))(x)
        
        x = AveragePooling1D(pool_size=3, strides=2, padding='same', name=f"average_pooling_2")(x)

        # Output batch normalization layer
        #outputs = BatchNormalization()(merged)
        outputs = x
        print(f"[DEBUG] Final Output shape: {outputs.shape}")

        # Build the encoder model
        self.encoder_model = Model(inputs=inputs, outputs=outputs, name="encoder")

        # Define the Adam optimizer with custom parameters
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],  # Set the learning rate
            beta_1=0.9,  # Default value
            beta_2=0.999,  # Default value
            epsilon=1e-7,  # Default value
            amsgrad=False  # Default value
        )

        self.encoder_model.compile(
            optimizer=adam_optimizer,
            loss=Huber(),
            metrics=['mse', 'mae'],
            run_eagerly=False  # Set to False for better performance unless debugging
        )
        print(f"[DEBUG] Encoder model compiled successfully.")





    def train(self, data):
        num_channels = data.shape[-1] if len(data.shape) > 2 else 1  # Get number of channels
        input_shape = data.shape[1]  # Get the input sequence length
        interface_size = self.params.get('interface_size', 4)  # Assuming interface size is in params

        # Reshape 2D data to 3D if necessary
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)  # Add a channel dimension

        # Rebuild the model with dynamic channel size
        self.configure_size(input_shape, interface_size, num_channels)

        # Now proceed with training
        print(f"Training encoder with data shape: {data.shape}")
        early_stopping = EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True)
        self.encoder_model.fit(data, data, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1, callbacks=[early_stopping], validation_split = 0.2)
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
    plugin.configure_size(input_shape=128, interface_size=4)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
