import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Concatenate, Conv1DTranspose, Reshape, LSTM, RepeatVector, TimeDistributed, Lambda, MultiHeadAttention # ADDED MultiHeadAttention
from keras.optimizers import Adam 

#Conv1D
from keras.layers import Conv1D
from keras.layers import LeakyReLU
from keras.initializers import HeNormal
# ADD: Imports for Positional Encoding, MHA block, and K backend
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.layers import LayerNormalization, Add 
import numpy as np # Already present, but ensure it's used for positional encoding helpers

# ADD: Positional Encoding helper functions (mirrored from encoder)
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class Plugin:
    """
    Plugin to define and manage a per-step generative network (decoder-like component)
    for a sequential, conditional generative model (e.g., CVAE for time series).
    This network takes the current latent variable z_t, RNN hidden state h_t (or h_{t-1}),
    and other conditions for step t, then outputs the generated data x'_t for the current step.
    Uses Conv1DTranspose layers. Output dimension is fixed to 6.
    """

    plugin_params = {
        "conv_activation": "relu",
        'learning_rate': 0.0001, 
        # Reference to encoder's parameters, will be overridden by config from AutoencoderManager
        "encoder_ref_initial_conv_filters": 128, 
        "encoder_ref_num_conv_layers": 4,
        "encoder_ref_num_strided_conv_layers": 2,
        "encoder_ref_min_conv_filters": 16,
        "conv_kernel_size": 5, # Kernel size for Conv1DTranspose
        "decoder_initial_seq_len": 4, # Initial sequence length after reshape
        "output_activation": "linear",
        "latent_dim": None,
        "rnn_hidden_dim": None,
        "conditioning_dim": None,
        "output_feature_dim": None, # MODIFIED: No longer fixed to 6, will be set by configure_model_architecture
    }

    plugin_debug_vars = [
        'latent_dim', 'rnn_hidden_dim', 'conditioning_dim', 'output_feature_dim',
        'encoder_ref_initial_conv_filters', 'encoder_ref_num_conv_layers', 
        'encoder_ref_num_strided_conv_layers', 'encoder_ref_min_conv_filters',
        'conv_kernel_size', 'decoder_initial_seq_len',
        'conv_activation', 'output_activation'
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.generative_network_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params: 
                self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_model_architecture(self, latent_dim: int, rnn_hidden_dim: int, 
                                     conditioning_dim: int, output_feature_dim: int, 
                                     config: dict = None):
        """
        Configures the per-step generative network with Conv1DTranspose.
        """
        if config is None:
            config = {}

        self.params['latent_dim'] = latent_dim
        self.params['rnn_hidden_dim'] = rnn_hidden_dim
        self.params['conditioning_dim'] = conditioning_dim
        self.params['output_feature_dim'] = output_feature_dim

        conv_activation_name = config.get("conv_activation", self.params.get("conv_activation", "relu"))
        output_activation_name = config.get("output_activation", self.params.get("output_activation", "linear"))
        
        # Get encoder structure to mirror it
        enc_initial_filters = config.get("initial_conv_filters", self.params.get("initial_conv_filters", 128))
        enc_num_conv_layers = config.get("num_conv_layers", self.params.get("num_conv_layers", 4))
        enc_num_strided_layers = config.get("num_strided_conv_layers", self.params.get("num_strided_conv_layers", 2))
        enc_min_filters = config.get("min_conv_filters", self.params.get("min_conv_filters", 16))
        conv_kernel_size = config.get("conv_kernel_size", self.params.get("conv_kernel_size", 5))
        window_size = config.get("window_size", 288)  # Get window size from config

        # Calculate the actual temporal dimension after encoder's conv layers
        encoder_output_temporal_dim = window_size
        for i in range(enc_num_conv_layers):
            stride = 2  # Each layer has stride=2
            encoder_output_temporal_dim = encoder_output_temporal_dim // stride
        
        print(f"[DEBUG DecoderPlugin] Calculated encoder output temporal dim: {encoder_output_temporal_dim}")
        
        # Calculate encoder filter progression to mirror it
        encoder_actual_output_filters = []
        encoder_actual_strides = []
        current_filters = enc_initial_filters
        
        for i in range(enc_num_conv_layers):
            encoder_actual_output_filters.append(current_filters)
            stride = 2  # FIXED: Always stride=2 to match encoder
            encoder_actual_strides.append(stride)
            if i < enc_num_conv_layers - 1:
                # Always halve filters to match encoder progression
                current_filters = max(enc_min_filters, current_filters // 2)

        # Reverse for decoder (transpose convolutions go from last to first)
        decoder_convt_output_filters = encoder_actual_output_filters[::-1]
        decoder_strides = encoder_actual_strides[::-1]

        print(f"[DEBUG DecoderPlugin] Configuring with: z_dim={latent_dim}, h_dim={rnn_hidden_dim}, cond_dim={conditioning_dim}, out_dim={output_feature_dim}")
        print(f"[DEBUG DecoderPlugin] Original window size: {window_size}, Encoder output temporal dim: {encoder_output_temporal_dim}")
        print(f"[DEBUG DecoderPlugin] Encoder structure ref: initial_filters={enc_initial_filters}, num_layers={enc_num_conv_layers}, num_strided={enc_num_strided_layers}")
        print(f"[DEBUG DecoderPlugin] Decoder ConvT output filters: {decoder_convt_output_filters}")
        print(f"[DEBUG DecoderPlugin] Decoder strides: {decoder_strides}")

        # MODIFIED: Define inputs for sequence processing using encoder's output temporal dimension
        input_z_seq = Input(shape=(encoder_output_temporal_dim, latent_dim), name="decoder_input_z_seq")
        input_h_context = Input(shape=(rnn_hidden_dim,), name="decoder_input_h_context") 
        input_conditions = Input(shape=(conditioning_dim,), name="decoder_input_conditions")

        # MODIFIED: Expand context and conditions to match encoder's output sequence length
        h_context_expanded = RepeatVector(encoder_output_temporal_dim, name="h_context_repeated")(input_h_context)
        conditions_expanded = RepeatVector(encoder_output_temporal_dim, name="conditions_repeated")(input_conditions)

        # MODIFIED: Concatenate all sequence inputs
        x = Concatenate(axis=-1, name="decoder_concat_seq")([input_z_seq, h_context_expanded, conditions_expanded])
        # x shape: (batch, encoder_output_temporal_dim, latent_dim + rnn_hidden_dim + conditioning_dim)

        # MODIFIED: Use Conv1DTranspose layers to mirror encoder's Conv1D layers
        # Let x be the output of the initial concatenation of z_seq, h_context_expanded, conditions_expanded
        # x shape: (batch, encoder_output_temporal_dim, L+H+C)
        
        # This loop upsamples x to (batch, window_size, enc_initial_filters)
        # and applies LeakyReLU
        for i in range(len(decoder_convt_output_filters)):
            stride = decoder_strides[i]
            filters = decoder_convt_output_filters[i]
            
            x = Conv1DTranspose(
                filters=filters,
                kernel_size=3,
                strides=stride,
                padding='same',
                activation=None,
                kernel_initializer=HeNormal(),
                name=f"decoder_conv1d_transpose_{i+1}"
            )(x)
            x = LeakyReLU(alpha=0.2, name=f"decoder_conv1d_transpose_{i+1}_leaky")(x)
            # The feature dimension of x after this loop is decoder_convt_output_filters[-1],
            # which is enc_initial_filters.
            # The temporal dimension is window_size (after potential extra upsampling).

        current_temporal_dim_after_conv_t = encoder_output_temporal_dim * (2 ** len(decoder_convt_output_filters))
        
        if current_temporal_dim_after_conv_t != window_size:
            missing_factor = window_size // current_temporal_dim_after_conv_t
            if missing_factor > 1 and (missing_factor & (missing_factor - 1)) == 0:
                extra_layer_filters = decoder_convt_output_filters[-1] # Should be enc_initial_filters
                x = Conv1DTranspose(
                    filters=extra_layer_filters,
                    kernel_size=3,
                    strides=missing_factor,
                    padding='same',
                    activation=None, 
                    kernel_initializer=HeNormal(),
                    name=f"decoder_conv1d_transpose_extra"
                )(x)
                x = LeakyReLU(alpha=0.2, name=f"decoder_conv1d_transpose_extra_leaky")(x)
        # At this point, x has shape (batch, window_size, enc_initial_filters)

        # --- ADD Positional Encoding before Late MultiHeadAttention ---
        # Get sequence length (window_size) and feature dimension (enc_initial_filters) for pos encoding
        # Using K.int_shape(x) is safer for symbolic tensors
        shape_x_before_mha = K.int_shape(x)
        seq_length_for_pos_enc = shape_x_before_mha[1] # Should be window_size
        feature_dim_for_pos_enc = shape_x_before_mha[2] # Should be enc_initial_filters
        
        if seq_length_for_pos_enc is None or feature_dim_for_pos_enc is None:
            # Fallback if symbolic shape is not fully defined, though it should be
            tf.print(f"[WARNING DecoderPlugin] Could not infer shape for positional encoding dynamically. Using configured window_size: {window_size} and enc_initial_filters: {enc_initial_filters}")
            seq_length_for_pos_enc = window_size
            feature_dim_for_pos_enc = enc_initial_filters

        pos_enc_decoder = positional_encoding(seq_length_for_pos_enc, feature_dim_for_pos_enc)
        x_pos_encoded = x + pos_enc_decoder
        # --- END Positional Encoding ---

        # --- Late MultiHeadAttention Block (mirrors encoder's MHA block structure) ---
        num_attention_heads_decoder = 2 # Consistent with encoder
        # Feature dimension of x_pos_encoded is feature_dim_for_pos_enc (enc_initial_filters)
        late_attn_key_dim = max(1, feature_dim_for_pos_enc // num_attention_heads_decoder)

        attention_output_decoder = MultiHeadAttention(
            num_heads=num_attention_heads_decoder,
            key_dim=late_attn_key_dim,
            
            name="late_self_attention"
        )(query=x_pos_encoded, value=x_pos_encoded, key=x_pos_encoded) # Use x_pos_encoded for Q, K, V
        
        # Add & Norm
        x_after_mha = Add(name="late_mha_add")([x_pos_encoded, attention_output_decoder])
        x_after_mha = LayerNormalization(name="late_mha_layernorm")(x_after_mha)
        # --- END Late MultiHeadAttention Block ---
        
        # Final Conv1D projection layer
        # Input to this layer is now x_after_mha
        final_projection_out = Conv1D(
            filters=output_feature_dim,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=output_activation_name,
            kernel_initializer=HeNormal(),
            name="final_conv_features"
        )(x_after_mha) # CHANGED: Input is now x_after_mha
        
        # Extract only the final time step to match 2D targets
        output_seq = Lambda(lambda t: t[:, -1, :], name="decoder_output_seq")(final_projection_out) # Input is final_projection_out
        # output_seq shape: (batch, output_feature_dim) - matches your 2D targets
        
        print(f"[DEBUG DecoderPlugin] Final output symbolic shape: {output_seq.shape}")
        expected_shape = (None, output_feature_dim)
        if output_seq.shape[1] != output_feature_dim:
            raise ValueError(f"Decoder output feature dimension {output_seq.shape[1]} does not match expected {output_feature_dim}")
        
        self.generative_network_model = Model(
            inputs=[input_z_seq, input_h_context, input_conditions],
            outputs=output_seq,
            name="sequence_conv_transpose_cvae_decoder"
        )
        
        print(f"[DEBUG DecoderPlugin] Sequence decoder model built successfully.")
        self.generative_network_model.summary(line_length=120)

    def train(self, *args, **kwargs):
        print("WARNING: DecoderPlugin.train() called. This component is trained as part of the larger CVAE model.")
        pass

    def decode(self, per_step_inputs: list):
        """
        Processes the inputs (z_t, h_t, conditions_t) to produce the generated output.

        Args:
            per_step_inputs (list): A list containing [z_t_batch, h_t_batch, conditions_t_batch].
        Returns:
            np.ndarray: The generated data x'_t_batch (batch_size, output_feature_dim).
        """
        if not self.generative_network_model:
            raise ValueError("Decoder model is not configured or loaded.")
        if not isinstance(per_step_inputs, list) or len(per_step_inputs) != 3:
            raise ValueError("per_step_inputs must be a list of three numpy arrays: [z_t_batch, h_t_batch, conditions_t_batch]")
        
        # print(f"[DecoderPlugin] Decoding with input shapes: {[data.shape for data in per_step_inputs]}")
        x_prime_t_batch = self.generative_network_model.predict(per_step_inputs, verbose=0)
        # print(f"[DecoderPlugin] Produced x_prime_t shape: {x_prime_t_batch.shape}")
        return x_prime_t_batch

    def save(self, file_path):
        if self.generative_network_model:
            # Use native Keras format (.keras) to avoid legacy HDF5 warnings
            if file_path.endswith('.h5'):
                tf.print(f"Warning: Converting legacy .h5 path to modern .keras format")
                file_path = file_path.replace('.h5', '.keras')
            self.generative_network_model.save(file_path, save_format='keras')
            print(f"Decoder model saved to {file_path}")
        else:
            print("Decoder model not available to save.")

    def load(self, file_path, compile_model=False):
        # Handle both legacy .h5 and modern .keras formats
        self.generative_network_model = tf.keras.models.load_model(file_path, compile=compile_model)
        print(f"Decoder model loaded from {file_path}")
        
        try: # Attempt to reconstruct key params from loaded model structure
            input_layers = self.generative_network_model.inputs
            self.params['latent_dim'] = input_layers[0].shape[-1]
            self.params['rnn_hidden_dim'] = input_layers[1].shape[-1]
            self.params['conditioning_dim'] = input_layers[2].shape[-1]
            
            output_layer = self.generative_network_model.outputs[0]
            self.params['output_feature_dim'] = output_layer.shape[-1] # This will now reflect the actual loaded model's output dim
            
            print(f"[DEBUG DecoderPlugin Load] Reconstructed params from model: "
                  f"z_dim={self.params['latent_dim']}, h_dim={self.params['rnn_hidden_dim']}, "
                  f"cond_dim={self.params['conditioning_dim']}, out_dim={self.params['output_feature_dim']}")
            # REMOVED: Warning about output_dim not being 6, as it's now configurable.
            # if self.params['output_feature_dim'] != 6:
            #      print(f"WARNING: Loaded decoder model has output_dim={self.params['output_feature_dim']}, expected 6.")
        except Exception as e:
            print(f"[WARNING DecoderPlugin Load] Could not fully reconstruct params from loaded model. Error: {e}.")

# Example of how this plugin might be used
if __name__ == "__main__":
    plugin = Plugin()
    
    _latent_dim = 32
    _h_dim = 64
    _cond_dim = 6 # Example: previous 6 target features for autoregression
    _output_dim_expected = 23 # MODIFIED for testing with new feature count
    
    # Example config override for testing
    test_config = {
        "initial_conv_filters": 128, # Matches encoder's initial for filter progression (128 -> 16 for encoder, 16 -> 128 for decoder ConvT)
        "conv_kernel_size": 5,
        "decoder_initial_seq_len": 4, # Example starting sequence length for upsampling
        "conv_activation": "relu", 
        "output_activation": "linear"
    }

    plugin.configure_model_architecture(
        latent_dim=_latent_dim,
        rnn_hidden_dim=_h_dim,
        conditioning_dim=_cond_dim,
        output_feature_dim=_output_dim_expected, # Pass the new expected output dim
        config=test_config
    )
    
    assert plugin.params['output_feature_dim'] == _output_dim_expected # MODIFIED assertion

    batch_size = 4
    dummy_z_t = np.random.rand(batch_size, _latent_dim).astype(np.float32)
    dummy_h_t = np.random.rand(batch_size, _h_dim).astype(np.float32)
    dummy_conditions_t = np.random.rand(batch_size, _cond_dim).astype(np.float32)
    
    x_prime_t = plugin.decode([dummy_z_t, dummy_h_t, dummy_conditions_t])
    
    print(f"\nTest decode output shape: x_prime_t: {x_prime_t.shape}")
    assert x_prime_t.shape == (batch_size, _output_dim_expected) # MODIFIED assertion

    model_path = "temp_cvae_decoder_network.keras"
    plugin.save(model_path)
    
    loaded_plugin = Plugin()
    loaded_plugin.load(model_path)
    
    x_prime_t_loaded = loaded_plugin.decode([dummy_z_t, dummy_h_t, dummy_conditions_t])
    print(f"Test loaded decode output shape: x_prime_t: {x_prime_t_loaded.shape}")
    assert x_prime_t_loaded.shape == (batch_size, _output_dim_expected) # MODIFIED assertion
    np.testing.assert_array_almost_equal(x_prime_t, x_prime_t_loaded)
    print(f"\nSave, load, and re-decode test successful for {_output_dim_expected}-feature output.") # MODIFIED print
    
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
