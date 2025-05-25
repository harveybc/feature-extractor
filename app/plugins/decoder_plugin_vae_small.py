import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Concatenate, Conv1DTranspose, Reshape, LSTM, RepeatVector, TimeDistributed, Lambda # ADDED RepeatVector, TimeDistributed, Lambda
from keras.optimizers import Adam 
from keras.regularizers import l2

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
        "l2_reg": 1e-5,
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
        'conv_activation', 'l2_reg', 'output_activation'
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
        for i in range(len(decoder_convt_output_filters)):
            stride = decoder_strides[i]
            filters = decoder_convt_output_filters[i]
            
            x = Conv1DTranspose(
                filters=filters,
                kernel_size=3,  # FIXED: Use kernel_size=3 to match encoder
                strides=stride,
                padding='same',
                activation=conv_activation_name,
                name=f"decoder_conv1d_transpose_{i+1}"
            )(x)
            
            # Debug: Print actual shape after each layer
            current_temporal_dim = encoder_output_temporal_dim * (2 ** (i + 1))
            print(f"[DEBUG DecoderPlugin] After Conv1DTranspose_{i+1}: stride={stride}, filters={filters}, expected temporal dim={current_temporal_dim}")

        # CRITICAL FIX: Ensure final output matches original window_size
        current_temporal_dim = encoder_output_temporal_dim * (2 ** len(decoder_convt_output_filters))
        if current_temporal_dim != window_size:
            print(f"[WARNING DecoderPlugin] Final temporal dim {current_temporal_dim} != window_size {window_size}")
            print(f"[WARNING DecoderPlugin] This will cause shape mismatch in training!")
            
            # Add additional upsampling layer if needed
            missing_factor = window_size // current_temporal_dim
            if missing_factor > 1 and (missing_factor & (missing_factor - 1)) == 0:  # Check if power of 2
                print(f"[FIX DecoderPlugin] Adding extra Conv1DTranspose with stride={missing_factor}")
                x = Conv1DTranspose(
                    filters=decoder_convt_output_filters[-1],  # Use same filters as last layer
                    kernel_size=3,
                    strides=missing_factor,
                    padding='same',
                    activation=conv_activation_name,
                    name=f"decoder_conv1d_transpose_extra"
                )(x)

        # SIMPLIFIED: Direct Conv1D output to final features, then extract last time step
        # Add final conv layer to get the right number of output features
        x = Conv1D(
            filters=output_feature_dim,
            kernel_size=1,  # 1x1 conv for feature transformation
            strides=1,
            padding='same',
            activation=output_activation_name,
            name="final_conv_features"
        )(x)  # shape: (batch, window_size, output_feature_dim)
        
        # Extract only the final time step to match 2D targets
        output_seq = Lambda(lambda x: x[:, -1, :], name="decoder_output_seq")(x)
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
            save_model(self.generative_network_model, file_path)
            print(f"Decoder model saved to {file_path}")
        else:
            print("Decoder model not available to save.")

    def load(self, file_path, compile_model=False): # Keras models are often loaded with compile=False if part of a larger system
        self.generative_network_model = load_model(file_path, compile=compile_model)
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
        "output_activation": "linear",
        "l2_reg": 1e-5
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
