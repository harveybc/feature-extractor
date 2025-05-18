import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Concatenate, Reshape, Conv1DTranspose, Flatten
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
        "output_feature_dim": 6, # Fixed
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
        Configures the per-step generative network with Conv1DTranspose. output_feature_dim will be forced to 6.
        """
        if config is None: config = {}

        self.params['latent_dim'] = latent_dim
        self.params['rnn_hidden_dim'] = rnn_hidden_dim
        self.params['conditioning_dim'] = conditioning_dim
        self.params['output_feature_dim'] = 6 # Enforce 6 output features

        if output_feature_dim != 6:
            print(f"WARNING: Decoder output_feature_dim passed as {output_feature_dim}, but overridden to 6.")

        # Get parameters, prioritizing external config, then self.params
        conv_activation_name = config.get("conv_activation", self.params.get("conv_activation", "relu"))
        l2_reg_val = config.get("l2_reg", self.params.get("l2_reg", 1e-5))
        conv_kernel_size_val = config.get("conv_kernel_size", self.params.get("conv_kernel_size", 5))
        decoder_initial_seq_len_val = config.get("decoder_initial_seq_len", self.params.get("decoder_initial_seq_len", 4))
        output_activation_name = config.get("output_activation", self.params.get("output_activation", "linear"))
        
        # Get encoder's structural parameters from the main config
        enc_initial_filters = config.get("initial_conv_filters", self.params.get("encoder_ref_initial_conv_filters"))
        enc_num_conv_layers = config.get("num_conv_layers", self.params.get("encoder_ref_num_conv_layers"))
        enc_num_strided_layers = config.get("num_strided_conv_layers", self.params.get("encoder_ref_num_strided_conv_layers"))
        enc_min_filters = config.get("min_conv_filters", self.params.get("encoder_ref_min_conv_filters"))

        # 1. Calculate encoder's actual output filter sizes and strides per layer
        encoder_actual_output_filters = []
        encoder_actual_strides = []
        current_enc_filters = enc_initial_filters
        for i in range(enc_num_conv_layers):
            encoder_actual_output_filters.append(current_enc_filters)
            is_strided_encoder_layer = (i < enc_num_strided_layers)
            encoder_actual_strides.append(2 if is_strided_encoder_layer else 1)
            
            if i < enc_num_conv_layers - 1:
                if is_strided_encoder_layer:
                    current_enc_filters = max(enc_min_filters, current_enc_filters // 2)
                else:
                    current_enc_filters = max(enc_min_filters, int(current_enc_filters * 0.8))
        
        # 2. Define Decoder's ConvT output filter progression (reversed from encoder's output filters)
        # Example: Encoder outputs [128, 64, 32, 16]. Decoder ConvT aims for [16, 32, 64, 128] as output filters.
        decoder_convt_output_filters = list(reversed(encoder_actual_output_filters))
        
        # The last ConvT layer's filters might be different (e.g., smaller, or related to output_feature_dim)
        # For now, let's keep it symmetric, so decoder_convt_output_filters[-1] would be enc_initial_filters.
        # Or, we can introduce a 'decoder_last_convt_filters' if needed.
        # Let's assume the last layer of decoder_convt_output_filters is the target.

        if not decoder_convt_output_filters:
            raise ValueError("Could not determine decoder ConvT filter progression. Check encoder parameters.")

        # 3. Initial Dense and Reshape
        # The first ConvT layer will take `decoder_convt_output_filters[0]` as its *input* channels.
        # This means the Reshape layer must output `decoder_convt_output_filters[0]` channels.
        first_convt_input_channels = decoder_convt_output_filters[0] # This is encoder_actual_output_filters[-1]
        
        dense_upsample_units = decoder_initial_seq_len_val * first_convt_input_channels

        print(f"[DEBUG DecoderPlugin] Configuring with: z_dim={latent_dim}, h_dim={rnn_hidden_dim}, cond_dim={conditioning_dim}, out_dim={self.params['output_feature_dim']}")
        print(f"[DEBUG DecoderPlugin] Encoder structure ref: initial_filters={enc_initial_filters}, num_layers={enc_num_conv_layers}, num_strided={enc_num_strided_layers}")
        print(f"[DEBUG DecoderPlugin] Encoder calculated output filters: {encoder_actual_output_filters}")
        print(f"[DEBUG DecoderPlugin] Encoder calculated strides: {encoder_actual_strides}")
        print(f"[DEBUG DecoderPlugin] Decoder ConvT target output filters: {decoder_convt_output_filters}")
        print(f"[DEBUG DecoderPlugin] Initial Dense units: {dense_upsample_units}, Reshape to: ({decoder_initial_seq_len_val}, {first_convt_input_channels})")

        # Define inputs for a single time step
        input_z_t = Input(shape=(latent_dim,), name="decoder_input_z_t")
        input_h_t = Input(shape=(rnn_hidden_dim,), name="decoder_input_h_t")
        input_conditions_t = Input(shape=(conditioning_dim,), name="decoder_input_conditions_t")

        # Concatenate all inputs
        concatenated_inputs = Concatenate(name="decoder_concat_inputs")([input_z_t, input_h_t, input_conditions_t])

        # Dense layer to prepare for reshaping
        x = Dense(
            units=dense_upsample_units,
            activation=conv_activation_name,
            kernel_regularizer=l2(l2_reg_val),
            name="decoder_dense_upsample_prep"
        )(concatenated_inputs)

        # Reshape for Conv1DTranspose
        x = Reshape(
            (decoder_initial_seq_len_val, first_convt_input_channels),
            name="decoder_reshape_for_conv"
        )(x)

        # Conv1DTranspose layers
        num_decoder_convt_layers = enc_num_conv_layers
        for i in range(num_decoder_convt_layers):
            # Output filters for this ConvT layer:
            # If decoder_convt_output_filters = [16, 32, 64, 128] (derived from encoder)
            # Layer 0 ConvT outputs 16 filters (already input shape)
            # Layer 1 ConvT outputs 32 filters
            # Layer 2 ConvT outputs 64 filters
            # Layer 3 ConvT outputs 128 filters
            # The `filters` param for Conv1DTranspose is its output channels.
            # The input channels for ConvT[i] is the output channels of ConvT[i-1]
            # (or from Reshape for i=0).
            
            # For ConvT layer `i`, its output filter count should be `decoder_convt_output_filters[i]`.
            # However, the list `decoder_convt_output_filters` was defined as `list(reversed(encoder_actual_output_filters))`.
            # So, `decoder_convt_output_filters[0]` is `encoder_actual_output_filters[-1]`.
            # `decoder_convt_output_filters[1]` is `encoder_actual_output_filters[-2]`.
            # This means the target output filters for ConvT[i] should be `decoder_convt_output_filters[i]` if we want to build up.
            # Let's use `target_output_filters_for_this_convT = decoder_convt_output_filters[i]`

            target_output_filters_for_this_convT = decoder_convt_output_filters[i]
            
            # Strides are reversed from encoder's strides
            current_convt_stride = encoder_actual_strides[enc_num_conv_layers - 1 - i]
            
            x = Conv1DTranspose(
                filters=target_output_filters_for_this_convT,
                kernel_size=conv_kernel_size_val,
                strides=current_convt_stride,
                padding='same',
                activation=conv_activation_name,
                kernel_regularizer=l2(l2_reg_val),
                name=f"decoder_conv_transpose_{i+1}"
            )(x)
        
        # Flatten the output of Conv1DTranspose layers
        x = Flatten(name="decoder_flatten_upsampled")(x)

        # Output layer for x'_t (6 features)
        output_x_prime_t = Dense(
            self.params['output_feature_dim'], 
            activation=output_activation_name, 
            name='decoder_output_x_prime_t', 
            kernel_regularizer=l2(l2_reg_val)
        )(x)

        self.generative_network_model = Model(
            inputs=[input_z_t, input_h_t, input_conditions_t],
            outputs=output_x_prime_t,
            name="dynamic_conv_transpose_cvae_decoder"
        )
        
        print(f"[DEBUG DecoderPlugin] Model built.")
        self.generative_network_model.summary(line_length=120)

    def train(self, *args, **kwargs):
        print("WARNING: DecoderPlugin.train() called. This component is trained as part of the larger CVAE model.")
        pass

    def decode(self, per_step_inputs: list):
        """
        Processes the inputs (z_t, h_t, conditions_t) to produce the generated 6-feature output.

        Args:
            per_step_inputs (list): A list containing [z_t_batch, h_t_batch, conditions_t_batch].
        Returns:
            np.ndarray: The generated data x'_t_batch (batch_size, 6).
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
            self.params['output_feature_dim'] = output_layer.shape[-1] # Should be 6
            
            print(f"[DEBUG DecoderPlugin Load] Reconstructed params from model: "
                  f"z_dim={self.params['latent_dim']}, h_dim={self.params['rnn_hidden_dim']}, "
                  f"cond_dim={self.params['conditioning_dim']}, out_dim={self.params['output_feature_dim']}")
            if self.params['output_feature_dim'] != 6:
                 print(f"WARNING: Loaded decoder model has output_dim={self.params['output_feature_dim']}, expected 6.")
        except Exception as e:
            print(f"[WARNING DecoderPlugin Load] Could not fully reconstruct params from loaded model. Error: {e}.")

# Example of how this plugin might be used
if __name__ == "__main__":
    plugin = Plugin()
    
    _latent_dim = 32
    _h_dim = 64
    _cond_dim = 6 # Example: previous 6 target features for autoregression
    _output_dim_expected = 6 
    
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
        output_feature_dim=_output_dim_expected, 
        config=test_config
    )
    
    assert plugin.params['output_feature_dim'] == 6

    batch_size = 4
    dummy_z_t = np.random.rand(batch_size, _latent_dim).astype(np.float32)
    dummy_h_t = np.random.rand(batch_size, _h_dim).astype(np.float32)
    dummy_conditions_t = np.random.rand(batch_size, _cond_dim).astype(np.float32)
    
    x_prime_t = plugin.decode([dummy_z_t, dummy_h_t, dummy_conditions_t])
    
    print(f"\nTest decode output shape: x_prime_t: {x_prime_t.shape}")
    assert x_prime_t.shape == (batch_size, 6)

    model_path = "temp_cvae_decoder_network.keras"
    plugin.save(model_path)
    
    loaded_plugin = Plugin()
    loaded_plugin.load(model_path)
    
    x_prime_t_loaded = loaded_plugin.decode([dummy_z_t, dummy_h_t, dummy_conditions_t])
    print(f"Test loaded decode output shape: x_prime_t: {x_prime_t_loaded.shape}")
    assert x_prime_t_loaded.shape == (batch_size, 6)
    np.testing.assert_array_almost_equal(x_prime_t, x_prime_t_loaded)
    print("\nSave, load, and re-decode test successful for 6-feature output.")
    
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
