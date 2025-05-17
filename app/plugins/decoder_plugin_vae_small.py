import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Concatenate, Reshape, Conv1DTranspose, Flatten # Removed BatchNormalization, LeakyReLU, Activation, Dropout
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
        "conv_activation": "relu",  # Activation for Dense layer before reshape and Conv1DTranspose layers
        'learning_rate': 0.0001, 
        "l2_reg": 1e-5,
        "initial_conv_filters": 128, # Should match the encoder's initial_conv_filters for filter progression
        "conv_kernel_size": 5,      # Kernel size for Conv1DTranspose, should match encoder's
        "decoder_initial_seq_len": 4, # Initial sequence length after reshape, before Conv1DTranspose
        "output_activation": "linear", # For the final Dense output layer
        # Parameters to be configured by AutoencoderManager:
        "latent_dim": None,         # Dimensionality of z_t
        "rnn_hidden_dim": None,     # Dimensionality of h_t (or h_{t-1})
        "conditioning_dim": None,   # Dimensionality of other conditions
        "output_feature_dim": 6,    # Fixed to 6
    }

    plugin_debug_vars = [
        'latent_dim', 'rnn_hidden_dim', 'conditioning_dim', 'output_feature_dim',
        'initial_conv_filters', 'conv_kernel_size', 'decoder_initial_seq_len',
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
        if config is None:
            config = {}

        self.params['latent_dim'] = latent_dim
        self.params['rnn_hidden_dim'] = rnn_hidden_dim
        self.params['conditioning_dim'] = conditioning_dim
        self.params['output_feature_dim'] = 6 # Enforce 6 output features

        if output_feature_dim != 6:
            print(f"WARNING: Decoder output_feature_dim passed as {output_feature_dim}, but overridden to 6.")

        # Get parameters, prioritizing external config, then self.params
        conv_activation_name = config.get("conv_activation", self.params.get("conv_activation", "relu"))
        l2_reg_val = config.get("l2_reg", self.params.get("l2_reg", 1e-5))
        
        # initial_conv_filters refers to the encoder's starting point, used to derive decoder filter progression
        # e.g., if encoder is 128->64->32->16, decoder ConvT is 16->32->64->128
        encoder_initial_conv_filters = config.get("initial_conv_filters", self.params.get("initial_conv_filters", 128))
        conv_kernel_size_val = config.get("conv_kernel_size", self.params.get("conv_kernel_size", 5))
        decoder_initial_seq_len_val = config.get("decoder_initial_seq_len", self.params.get("decoder_initial_seq_len", 4))
        output_activation_name = config.get("output_activation", self.params.get("output_activation", "linear"))
        
        final_output_dim = self.params['output_feature_dim'] # Should be 6

        # Define filter progression for Conv1DTranspose (mirrors encoder's reduction)
        # Encoder: F, F/2, F/4, F/8. Decoder ConvT: F/8, F/4, F/2, F.
        decoder_conv_filters_progression = [
            encoder_initial_conv_filters // 8,
            encoder_initial_conv_filters // 4,
            encoder_initial_conv_filters // 2,
            encoder_initial_conv_filters
        ]
        # Ensure filters are at least 1
        decoder_conv_filters_progression = [max(1, f) for f in decoder_conv_filters_progression]
        
        dense_upsample_units = decoder_initial_seq_len_val * decoder_conv_filters_progression[0]

        print(f"[DEBUG DecoderPlugin] Configuring with: z_dim={latent_dim}, h_dim={rnn_hidden_dim}, cond_dim={conditioning_dim}, out_dim={final_output_dim}")
        print(f"[DEBUG DecoderPlugin] ConvT Params: initial_seq_len={decoder_initial_seq_len_val}, base_filters_for_dense={decoder_conv_filters_progression[0]}, "
              f"dense_upsample_units={dense_upsample_units}, kernel_size={conv_kernel_size_val}, activation={conv_activation_name}")
        print(f"[DEBUG DecoderPlugin] ConvT Filter Progression: {decoder_conv_filters_progression}")
        print(f"[DEBUG DecoderPlugin] L2: {l2_reg_val}, Output Activation: {output_activation_name}")

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
            (decoder_initial_seq_len_val, decoder_conv_filters_progression[0]),
            name="decoder_reshape_for_conv"
        )(x)

        # Conv1DTranspose layers
        for i, num_filters in enumerate(decoder_conv_filters_progression):
            # The first layer uses decoder_conv_filters_progression[0] which is already set by Reshape
            # So, Conv1DTranspose layers will effectively use filters from index 0 to 3 for their output channels
            # The actual number of filters for Conv1DTranspose[i] is decoder_conv_filters_progression[i]
            x = Conv1DTranspose(
                filters=num_filters, # This is the output filter count for this layer
                kernel_size=conv_kernel_size_val,
                strides=2,
                padding='same',
                activation=conv_activation_name,
                kernel_regularizer=l2(l2_reg_val),
                name=f"decoder_conv_transpose_{i+1}"
            )(x)
            # After this layer, num_filters is num_filters, seq_len is doubled.

        # Flatten the output of Conv1DTranspose layers
        x = Flatten(name="decoder_flatten_upsampled")(x)

        # Output layer for x'_t (6 features)
        output_x_prime_t = Dense(
            final_output_dim, 
            activation=output_activation_name, 
            name='decoder_output_x_prime_t', 
            kernel_regularizer=l2(l2_reg_val)
        )(x)

        self.generative_network_model = Model(
            inputs=[input_z_t, input_h_t, input_conditions_t],
            outputs=output_x_prime_t,
            name="conv_transpose_cvae_decoder"
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
