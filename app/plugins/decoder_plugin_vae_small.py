import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Concatenate, BatchNormalization, LeakyReLU, Activation, Dropout
from keras.optimizers import Adam # Optimizer is usually for the main model
from tensorflow.keras.initializers import GlorotUniform, HeNormal # Keras uses these by default if not specified
from keras.regularizers import l2

class Plugin:
    """
    Plugin to define and manage a per-step generative network (decoder-like component)
    for a sequential, conditional generative model (e.g., CVAE for time series).
    This network takes the current latent variable z_t, RNN hidden state h_t (or h_{t-1}),
    and other conditions for step t, then outputs the generated data x'_t for the current step.
    The output dimension will be configured to 6 for OHLC + 2 derived features.
    """

    plugin_params = {
        "dense_activation": "relu",  # Common for intermediate dense layers
        'learning_rate': 0.0001, # Example, part of the larger model's optimizer
        "l2_reg": 1e-5,
        "dense_layer_sizes": [128, 64], # Example intermediate layer sizes
        "use_batch_norm_dense": True, # Option to use Batch Norm in dense layers
        "dropout_rate_dense": 0.1, # Option for Dropout in dense layers
        "output_activation": "linear", # 'linear' for regression, 'sigmoid'/'tanh' if data is bounded
        # Parameters to be configured by AutoencoderManager:
        "latent_dim": None,         # Dimensionality of z_t
        "rnn_hidden_dim": None,     # Dimensionality of h_t (or h_{t-1})
        "conditioning_dim": None,   # Dimensionality of other conditions (e.g., previous 6 target features)
        "output_feature_dim": 6,    # Fixed to 6 for OHLC + 2 derived features
    }

    plugin_debug_vars = [
        'latent_dim', 'rnn_hidden_dim', 'conditioning_dim', 'output_feature_dim',
        'dense_layer_sizes', 'dense_activation', 'l2_reg', 'output_activation',
        'use_batch_norm_dense', 'dropout_rate_dense'
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        # Ensure mutable defaults are copied correctly if any were complex
        if "dense_layer_sizes" in self.plugin_params: # Example for list
            self.params["dense_layer_sizes"] = list(self.plugin_params["dense_layer_sizes"])
        self.generative_network_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params: # Only update known params
                self.params[key] = value
            # Deep copy for mutable types if they are directly set
            if key == "dense_layer_sizes" and isinstance(value, list):
                self.params[key] = list(value)


    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_model_architecture(self, latent_dim: int, rnn_hidden_dim: int, 
                                     conditioning_dim: int, output_feature_dim: int, 
                                     config: dict = None):
        """
        Configures the per-step generative network. output_feature_dim will be forced to 6.

        Args:
            latent_dim (int): Dimensionality of the input latent variable z_t.
            rnn_hidden_dim (int): Dimensionality of the RNN's hidden state h_t (or h_{t-1}) input.
            conditioning_dim (int): Dimensionality of other concatenated conditioning variables for step t.
            output_feature_dim (int): Expected to be 6 for this decoder. Will be overridden if different.
            config (dict, optional): External configuration dictionary to override plugin_params.
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
        dense_activation_name = config.get("dense_activation", self.params.get("dense_activation", "relu"))
        l2_reg_val = config.get("l2_reg", self.params.get("l2_reg", 1e-5))
        dense_layer_sizes = config.get("dense_layer_sizes", self.params.get("dense_layer_sizes", [128, 64]))
        use_batch_norm_dense = config.get("use_batch_norm_dense", self.params.get("use_batch_norm_dense", True))
        dropout_rate_dense = config.get("dropout_rate_dense", self.params.get("dropout_rate_dense", 0.1))
        output_activation_name = config.get("output_activation", self.params.get("output_activation", "linear"))
        
        final_output_dim = self.params['output_feature_dim'] # Should be 6

        print(f"[DEBUG DecoderPlugin] Configuring with: z_dim={latent_dim}, h_dim={rnn_hidden_dim}, cond_dim={conditioning_dim}, out_dim={final_output_dim}")
        print(f"[DEBUG DecoderPlugin] Dense layers: {dense_layer_sizes}, Activation: {dense_activation_name}, L2: {l2_reg_val}, Output Activation: {output_activation_name}")
        print(f"[DEBUG DecoderPlugin] Use BN Dense: {use_batch_norm_dense}, Dropout Dense: {dropout_rate_dense}")

        # Define inputs for a single time step
        input_z_t = Input(shape=(latent_dim,), name="decoder_input_z_t")
        input_h_t = Input(shape=(rnn_hidden_dim,), name="decoder_input_h_t")
        input_conditions_t = Input(shape=(conditioning_dim,), name="decoder_input_conditions_t")

        # Concatenate all inputs
        concatenated_inputs = Concatenate(name="decoder_concat_inputs")([input_z_t, input_h_t, input_conditions_t])

        # Intermediate Dense layers
        x = concatenated_inputs
        for i, units in enumerate(dense_layer_sizes):
            x = Dense(
                units,
                kernel_regularizer=l2(l2_reg_val),
                # kernel_initializer=HeNormal() if dense_activation_name in ['relu', 'leakyrelu'] else GlorotUniform(), # Example explicit initializer
                name=f"decoder_dense_layer_{i+1}"
            )(x)
            if use_batch_norm_dense:
                x = BatchNormalization(name=f"decoder_bn_dense_{i+1}")(x)
            
            if dense_activation_name.lower() == 'leakyrelu':
                x = LeakyReLU(name=f"decoder_leakyrelu_dense_{i+1}")(x)
            else:
                x = Activation(dense_activation_name, name=f"decoder_activation_dense_{i+1}")(x)
            
            if dropout_rate_dense > 0:
                x = Dropout(dropout_rate_dense, name=f"decoder_dropout_dense_{i+1}")(x)

        # Output layer for x'_t (6 features)
        output_x_prime_t = Dense(final_output_dim, activation=output_activation_name, name='decoder_output_x_prime_t', kernel_regularizer=l2(l2_reg_val))(x)

        self.generative_network_model = Model(
            inputs=[input_z_t, input_h_t, input_conditions_t],
            outputs=output_x_prime_t,
            name="cvae_decoder_network"
        )
        
        print(f"[DEBUG DecoderPlugin] Model built.")
        self.generative_network_model.summary()

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
        "dense_layer_sizes": [100, 50], 
        "dense_activation": "elu", 
        "output_activation": "linear", # Assuming unscaled output for prices
        "use_batch_norm_dense": True,
        "dropout_rate_dense": 0.05
    }

    plugin.configure_model_architecture(
        latent_dim=_latent_dim,
        rnn_hidden_dim=_h_dim,
        conditioning_dim=_cond_dim,
        output_feature_dim=_output_dim_expected, # This will be forced to 6 internally
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
