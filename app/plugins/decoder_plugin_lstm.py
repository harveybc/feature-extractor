import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, UpSampling1D, Reshape, Flatten, Conv1DTranspose,Dropout
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from keras.models import Model, load_model, save_model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, MaxPooling1D, Cropping1D, LeakyReLU,Input
import math
from tensorflow.keras.layers import ZeroPadding1D
from keras.layers import Add, LayerNormalization, AveragePooling1D, Bidirectional, LSTM
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras import backend as K

from keras.layers import Input
import tensorflow as tf
from tensorflow.keras.losses import Huber
# TimeDistributed
from keras.layers import TimeDistributed


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
    plugin_params = {
        'intermediate_layers': 3, 
        'learning_rate': 0.00002,
        'dropout_rate': 0.001,
    }

    plugin_debug_vars = ['interface_size', 'output_shape', 'intermediate_layers']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value 

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_size(self, interface_size, output_shape, num_channels, encoder_output_shape, use_sliding_windows, config=None):
        print(f"[DEBUG] Starting decoder configuration with interface_size={interface_size}, output_shape={output_shape}, num_channels={num_channels}, encoder_output_shape={encoder_output_shape}, use_sliding_windows={use_sliding_windows}")
        
        self.params['interface_size'] = interface_size
        self.params['output_shape'] = output_shape

        sequence_length, num_filters = encoder_output_shape
        print(f"[DEBUG] Extracted sequence_length={sequence_length}, num_filters={num_filters} from encoder_output_shape.")

        num_intermediate_layers = self.params['intermediate_layers']
        print(f"[DEBUG] Number of intermediate layers={num_intermediate_layers}")
        
        layers = [output_shape*2]
        current_size = output_shape*2
        l2_reg = 1e-2
        for i in range(num_intermediate_layers-1):
            next_size = current_size // 2
            if next_size < interface_size:
                next_size = interface_size
            layers.append(next_size)
            current_size = next_size

        layers.append(interface_size)

        layer_sizes = layers[::-1]
        print(f"[DEBUG] Calculated decoder layer sizes: {layer_sizes}")

        window_size =config.get("window_size", 288)
        merged_units = config.get("initial_layer_size", 128)
        branch_units = merged_units//config.get("layer_size_divisor", 2)
        lstm_units = branch_units//config.get("layer_size_divisor", 2)
        activation = config.get("activation", "tanh")
        l2_reg = config.get("l2_reg", self.params.get("l2_reg", 1e-6))
        num_attention_heads = 2

        # Assume the following variables are defined in the outer scope:
        # window_size, branch_units (should typically be 2*lstm_units if encoder ends with BiLSTM), 
        # lstm_units, num_attention_heads, l2_reg, num_channels
        # Also assume the function positional_encoding(seq_length, feature_dim) is defined.

        # --- Decoder input (latent) ---
        # IMPORTANT: Verify that shape=((window_size // 4), branch_units) EXACTLY matches 
        # the actual output shape (excluding batch dim) of your encoder model. 
        # The sequence length (window_size // 4) is only accurate if encoder pooling used 
        # padding='same' or if 'valid' padding resulted in exactly this length.
        # 'branch_units' must match the feature dimension of the encoder's output.
        decoder_input = Input(
            shape=((window_size // 4) , branch_units), # VERIFY THIS SHAPE!
            name="decoder_input"
        )
        x = decoder_input

        # --- Inverse of AveragePooling1D_2 ---
        # Upsample the sequence length
        x = UpSampling1D(size=2, name="decoder_upsampling_1")(x)

        # --- Inverse of feature_lstm_2 ---
        # Added kernel_regularizer back for consistency with typical encoder setup
        x = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg)), # Added regularizer
                        name="decoder_lstm_1")(x)

        # --- Inverse of feature_lstm_1 ---
        # Added kernel_regularizer back for consistency with typical encoder setup
        x = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg)), # Added regularizer
                        name="decoder_lstm_2")(x)

        # --- Inverse of AveragePooling1D_1 ---
        # Upsample the sequence length again
        x = UpSampling1D(size=2, name="decoder_upsampling_2")(x)
        # Note: Sequence length after this layer might not be exactly window_size if 
        # encoder used padding='valid'. If convergence fails, consider using padding='same' 
        # in encoder's pooling layers or adding a Cropping1D/ZeroPadding1D layer here.
        current_seq_len = K.int_shape(x)[1]
        print(f"[DEBUG] Decoder sequence length after UpSampling_2: {current_seq_len}") # Debug print

        # --- Inverse of Self-Attention Block 1 ---
        # Add positional encoding before attention, mirroring the encoder
        last_layer_shape = K.int_shape(x)
        feature_dim_attn = last_layer_shape[-1] # Should be 2 * lstm_units from BiLSTM output
        seq_length_attn = last_layer_shape[1]   # Current sequence length

        # --- Add Positional Encoding ---
        # Ensure the positional_encoding function is defined and accessible
        # This mirrors the encoder structure where positional encoding is added before Attention
        try:
            # Check if positional_encoding function is available and seq_length_attn is not None
            if callable(positional_encoding) and seq_length_attn is not None:
                pos_enc = positional_encoding(seq_length_attn, feature_dim_attn)
                print(f"[DEBUG] Decoder: Adding positional encoding for seq_len={seq_length_attn}, dim={feature_dim_attn}")
                x = Add(name="decoder_add_pos_enc")([x, pos_enc])
            else:
                print(f"[WARN] Decoder: Positional encoding function not callable or seq_length_attn is None. Skipping.")
        except NameError:
            print(f"[WARN] Decoder: positional_encoding function not defined. Skipping.")


        # --- Attention Mechanism ---
        # Define key dimension for attention based on current feature dimension
        # Check if feature_dim_attn is not None before division
        if feature_dim_attn is not None and num_attention_heads > 0:
            attention_key_dim = feature_dim_attn // num_attention_heads
            print(f"[DEBUG] Decoder Attention Key Dim: {attention_key_dim}") # Debug print

            # Apply MultiHeadAttention (self-attention on the decoder sequence)
            # Added kernel_regularizer back for consistency
            attention_output = MultiHeadAttention(
                num_heads=num_attention_heads,
                key_dim=attention_key_dim,
                kernel_regularizer=l2(l2_reg), # Added regularizer
                name="decoder_multihead_attention_1"
            )(query=x, value=x, key=x) # Self-attention
            x_res = Add(name="decoder_add_attention_residual")([x, attention_output]) # Residual connection
            x = LayerNormalization(name="decoder_layernorm_attention")(x_res)
        else:
            print(f"[WARN] Decoder: Skipping Attention Block due to invalid feature_dim_attn ({feature_dim_attn}) or num_attention_heads ({num_attention_heads}).")
            # If attention is skipped, x remains the output of UpSampling1D/PositionalEncoding


        # --- Output Layer ---
        # Project back to the original number of channels for each time step
        # Use 'linear' activation for reconstruction, unless input was scaled differently
        # This layer ensures the output has the shape (batch, sequence_length_after_upsampling, num_channels)
        outputs = TimeDistributed(Dense(num_channels, activation='linear', name="output_dense"),
                                name="decoder_output_projection")(x)

        # Final check on the output shape AFTER the model is built might be more reliable
        print(f"[DEBUG] Decoder: Defined final output layer targeting num_channels={num_channels}.")
        # Note: The actual output shape depends on the input shape and layer operations. 
        # Verify model.summary() for the final output shape. E.g., decoder_model.summary()

        # Build the encoder model
        self.model = Model(inputs=decoder_input, outputs=outputs, name="decoder")
        
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-2,
            amsgrad=False
        )

        self.model.compile(
            optimizer=adam_optimizer,
            loss=Huber(),
            metrics=['mse', 'mae'],
            run_eagerly=False
        )
        print(f"[DEBUG] Model compiled successfully.")




    def train(self, encoded_data, original_data,config):
        encoded_data = encoded_data.reshape((encoded_data.shape[0], -1))
        original_data = original_data.reshape((original_data.shape[0], -1))
        early_stopping = EarlyStopping(monitor='val_mae', patience=25, restore_best_weights=True, verbose=1)
        self.model.fit(encoded_data, original_data, epochs=self.params['epochs'], batch_size=config.get("batch_size"), verbose=1, callbacks=[early_stopping],validation_split = 0.2)

    def decode(self, encoded_data, use_sliding_windows, original_feature_size):
        print(f"[decode] Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.model.predict(encoded_data)

        if not use_sliding_windows:
            # Reshape to match the original feature size if not using sliding windows
            decoded_data = decoded_data.reshape((decoded_data.shape[0], original_feature_size))
            print(f"[decode] Reshaped decoded data to match original feature size: {decoded_data.shape}")
        else:
            print(f"[decode] Decoded data shape: {decoded_data.shape}")

        return decoded_data

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = load_model(file_path)

    def calculate_mse(self, original_data, reconstructed_data):
        original_data = original_data.reshape((original_data.shape[0], -1))
        reconstructed_data = reconstructed_data.reshape((original_data.shape[0], -1))
        mse = np.mean(np.square(original_data - reconstructed_data))
        return mse

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.configure_size(interface_size=4, output_shape=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
