import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Concatenate, Conv1DTranspose, Reshape, LSTM, RepeatVector, TimeDistributed # ADDED RepeatVector, TimeDistributed
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

        # Calculate encoder filter progression to mirror it
        encoder_actual_output_filters = []
        encoder_actual_strides = []
        current_filters = enc_initial_filters
        
        for i in range(enc_num_conv_layers):
            encoder_actual_output_filters.append(current_filters)
            stride = 2 if i < enc_num_strided_layers else 1
            encoder_actual_strides.append(stride)
            if i < enc_num_conv_layers - 1:
                if stride == 2:
                    current_filters = max(enc_min_filters, current_filters // 2)
                else:
                    current_filters = max(enc_min_filters, int(current_filters * 0.8))

        # Reverse for decoder (transpose convolutions go from last to first)
        decoder_convt_output_filters = encoder_actual_output_filters[::-1]
        decoder_strides = encoder_actual_strides[::-1]

        print(f"[DEBUG DecoderPlugin] Configuring with: z_dim={latent_dim}, h_dim={rnn_hidden_dim}, cond_dim={conditioning_dim}, out_dim={output_feature_dim}")
        print(f"[DEBUG DecoderPlugin] Window size: {window_size}")
        print(f"[DEBUG DecoderPlugin] Encoder structure ref: initial_filters={enc_initial_filters}, num_layers={enc_num_conv_layers}, num_strided={enc_num_strided_layers}")
        print(f"[DEBUG DecoderPlugin] Decoder ConvT output filters: {decoder_convt_output_filters}")
        print(f"[DEBUG DecoderPlugin] Decoder strides: {decoder_strides}")

        # MODIFIED: Define inputs for sequence processing instead of single time step
        input_z_seq = Input(shape=(window_size, latent_dim), name="decoder_input_z_seq")
        input_h_context = Input(shape=(rnn_hidden_dim,), name="decoder_input_h_context") 
        input_conditions = Input(shape=(conditioning_dim,), name="decoder_input_conditions")

        # MODIFIED: Expand context and conditions to match sequence length
        h_context_expanded = RepeatVector(window_size, name="h_context_repeated")(input_h_context)
        conditions_expanded = RepeatVector(window_size, name="conditions_repeated")(input_conditions)

        # MODIFIED: Concatenate all sequence inputs
        x = Concatenate(axis=-1, name="decoder_concat_seq")([input_z_seq, h_context_expanded, conditions_expanded])
        # x shape: (batch, window_size, latent_dim + rnn_hidden_dim + conditioning_dim)

        # MODIFIED: Use Conv1DTranspose layers to mirror encoder's Conv1D layers
        for i in range(len(decoder_convt_output_filters)):
            stride = decoder_strides[i]
            filters = decoder_convt_output_filters[i]
            
            x = Conv1DTranspose(
                filters=filters,
                kernel_size=conv_kernel_size,
                strides=stride,
                padding='same',
                activation=conv_activation_name,
                # NO kernel_regularizer as requested
                name=f"decoder_conv1d_transpose_{i+1}"
            )(x)

        # MODIFIED: Final output layer using TimeDistributed
        output_seq = TimeDistributed(
            Dense(output_feature_dim, activation=output_activation_name, name="final_dense"),
            name="decoder_output_seq"
        )(x)  # shape: (batch, window_size, output_feature_dim)

        self.generative_network_model = Model(
            inputs=[input_z_seq, input_h_context, input_conditions],
            outputs=output_seq,
            name="sequence_conv_transpose_cvae_decoder"
        )
        
        print(f"[DEBUG DecoderPlugin] Sequence decoder model built.")
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
