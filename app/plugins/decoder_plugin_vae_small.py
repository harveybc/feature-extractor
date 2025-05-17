import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Concatenate # Changed imports
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
# Removed unused imports like Conv1DTranspose, Reshape, Cropping1D, ZeroPadding1D etc.
# from keras.callbacks import EarlyStopping # Callbacks are usually for training the whole model
# from tensorflow.keras.losses import Huber


class Plugin:
    """
    Plugin to define and manage a per-step generative network (decoder-like component)
    for a sequential, conditional generative model (e.g., VRNN).
    This network takes the current latent variable z_t, RNN hidden state h_t (or h_{t-1}),
    and other conditions for step t, then outputs the generated data x'_t for the current step.
    """

    plugin_params = {
        "activation": "relu",  # Common for intermediate dense layers
        'learning_rate': 0.0001, # Example, will be part of the larger model's optimizer
        "l2_reg": 1e-5,
        "dense_layer_sizes": [64, 128], # Example intermediate layer sizes
        "output_activation": "linear", # Or 'sigmoid'/'tanh' if data is bounded and scaled
        # Parameters to be configured:
        "latent_dim": None,         # Dimensionality of z_t
        "rnn_hidden_dim": None,     # Dimensionality of h_t (or h_{t-1} if used as direct input)
        "conditioning_dim": None,   # Dimensionality of other conditions (seasonal, current fundamentals, history)
        "output_feature_dim": None, # Dimensionality of x'_t (e.g., 6 base features)
    }

    plugin_debug_vars = [
        'latent_dim', 'rnn_hidden_dim', 'conditioning_dim', 'output_feature_dim',
        'dense_layer_sizes', 'activation', 'l2_reg', 'output_activation'
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.generative_network_model = None # Renamed from model
        # Attributes like 'interface_size_config', 'output_shape_config', etc. from old VAE decoder are no longer directly applicable in the same way.

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value # Allow setting any, or check against self.params.keys()

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_model_architecture(self, latent_dim: int, rnn_hidden_dim: int, conditioning_dim: int, output_feature_dim: int, config: dict = None):
        """
        Configures the per-step generative network.

        Args:
            latent_dim (int): Dimensionality of the input latent variable z_t.
            rnn_hidden_dim (int): Dimensionality of the RNN's hidden state h_t (or h_{t-1}) input.
            conditioning_dim (int): Dimensionality of other concatenated conditioning variables for step t.
            output_feature_dim (int): Desired dimensionality for the output data x'_t (e.g., 6 base features).
            config (dict, optional): External configuration dictionary to override plugin_params.
        """
        if config is None:
            config = {}

        self.params['latent_dim'] = latent_dim
        self.params['rnn_hidden_dim'] = rnn_hidden_dim
        self.params['conditioning_dim'] = conditioning_dim
        self.params['output_feature_dim'] = output_feature_dim

        # Get parameters, prioritizing external config, then self.params
        activation = config.get("activation", self.params.get("activation", "relu"))
        l2_reg_val = config.get("l2_reg", self.params.get("l2_reg", 1e-5))
        dense_layer_sizes = config.get("dense_layer_sizes", self.params.get("dense_layer_sizes", [64, 128]))
        output_activation = config.get("output_activation", self.params.get("output_activation", "linear"))
        
        print(f"[DEBUG PerStepGenerativeNetwork] Configuring with: z_dim={latent_dim}, h_dim={rnn_hidden_dim}, cond_dim={conditioning_dim}, out_dim={output_feature_dim}")
        print(f"[DEBUG PerStepGenerativeNetwork] Dense layers: {dense_layer_sizes}, Activation: {activation}, L2: {l2_reg_val}, Output Activation: {output_activation}")

        # Define inputs for a single time step
        input_z_t = Input(shape=(latent_dim,), name="input_z_t")
        input_h_t = Input(shape=(rnn_hidden_dim,), name="input_h_t") # Current/previous RNN hidden state
        input_conditions_t = Input(shape=(conditioning_dim,), name="input_conditions_t")

        # Concatenate all inputs
        concatenated_inputs = Concatenate(name="concat_inputs_gen_net")([input_z_t, input_h_t, input_conditions_t])

        # Intermediate Dense layers
        x = concatenated_inputs
        for i, units in enumerate(dense_layer_sizes):
            x = Dense(
                units,
                activation=activation,
                kernel_regularizer=l2(l2_reg_val),
                name=f"gen_dense_layer_{i+1}"
            )(x)

        # Output layer for x'_t
        output_x_prime_t = Dense(output_feature_dim, activation=output_activation, name='output_x_prime_t', kernel_regularizer=l2(l2_reg_val))(x)

        self.generative_network_model = Model(
            inputs=[input_z_t, input_h_t, input_conditions_t],
            outputs=output_x_prime_t,
            name="per_step_generative_network"
        )
        
        print(f"[DEBUG PerStepGenerativeNetwork] Model built. Inputs: {[input_z_t, input_h_t, input_conditions_t]}, Outputs: {output_x_prime_t}")
        self.generative_network_model.summary()

    def train(self, *args, **kwargs):
        # This plugin defines a component. Training is handled by the main sequential model
        # (e.g., the VRNN model in synthetic-datagen's OptimizerPlugin).
        print("WARNING: PerStepGenerativeNetworkPlugin.train() called. This component is typically trained as part of a larger sequential model.")
        pass

    def decode(self, per_step_inputs: list): # Renamed from 'decode' for clarity, though 'generate_step' might be better
        """
        Processes the inputs for a single time step (or a batch of time steps)
        to produce the generated data x'_t.

        Args:
            per_step_inputs (list): A list containing [z_t_batch, h_t_batch, conditions_t_batch].
                                   Shapes:
                                   - z_t_batch: (batch_size, latent_dim)
                                   - h_t_batch: (batch_size, rnn_hidden_dim)
                                   - conditions_t_batch: (batch_size, conditioning_dim)
        Returns:
            np.ndarray: The generated data x'_t_batch for the current step.
        """
        if not self.generative_network_model:
            raise ValueError("Per-step generative network model is not configured or loaded.")
        if not isinstance(per_step_inputs, list) or len(per_step_inputs) != 3:
            raise ValueError("per_step_inputs must be a list of three numpy arrays: [z_t_batch, h_t_batch, conditions_t_batch]")
        
        print(f"[PerStepGenerativeNetwork] Generating step with input shapes: {[data.shape for data in per_step_inputs]}")
        x_prime_t_batch = self.generative_network_model.predict(per_step_inputs)
        print(f"[PerStepGenerativeNetwork] Produced x_prime_t shape: {x_prime_t_batch.shape}")
        return x_prime_t_batch

    def save(self, file_path):
        if self.generative_network_model:
            save_model(self.generative_network_model, file_path)
            print(f"Per-step generative network model saved to {file_path}")
        else:
            print("Per-step generative network model not available to save.")

    def load(self, file_path, compile_model=False):
        self.generative_network_model = load_model(file_path, compile=compile_model)
        print(f"Per-step generative network model loaded from {file_path}")
        
        try:
            input_layers = self.generative_network_model.inputs
            self.params['latent_dim'] = input_layers[0].shape[-1]
            self.params['rnn_hidden_dim'] = input_layers[1].shape[-1]
            self.params['conditioning_dim'] = input_layers[2].shape[-1]
            
            output_layer = self.generative_network_model.outputs[0] # Assuming single output
            self.params['output_feature_dim'] = output_layer.shape[-1]
            
            print(f"[DEBUG PerStepGenerativeNetwork Load] Reconstructed params from model: "
                  f"z_dim={self.params['latent_dim']}, h_dim={self.params['rnn_hidden_dim']}, "
                  f"cond_dim={self.params['conditioning_dim']}, out_dim={self.params['output_feature_dim']}")
        except Exception as e:
            print(f"[WARNING PerStepGenerativeNetwork Load] Could not fully reconstruct params from loaded model structure. Error: {e}. Ensure manual configuration if needed.")

# Example of how this plugin might be used (conceptual)
if __name__ == "__main__":
    plugin = Plugin()
    
    # Example dimensions
    _latent_dim = 32
    _h_dim = 64
    _cond_dim = 10 # e.g., 6 seasonal + 2 current fundamentals + 2 history summary
    _output_dim = 6 # Number of base features to generate
    
    plugin.configure_model_architecture(
        latent_dim=_latent_dim,
        rnn_hidden_dim=_h_dim,
        conditioning_dim=_cond_dim,
        output_feature_dim=_output_dim,
        config={"dense_layer_sizes": [80, 100], "activation": "elu", "output_activation": "sigmoid"}
    )
    
    # Create dummy batch data for testing decode/generate_step
    batch_size = 4
    dummy_z_t = np.random.rand(batch_size, _latent_dim).astype(np.float32)
    dummy_h_t = np.random.rand(batch_size, _h_dim).astype(np.float32)
    dummy_conditions_t = np.random.rand(batch_size, _cond_dim).astype(np.float32)
    
    x_prime_t = plugin.decode([dummy_z_t, dummy_h_t, dummy_conditions_t])
    
    print(f"\nTest decode output shape: x_prime_t: {x_prime_t.shape}")

    # Test save and load
    model_path = "temp_per_step_generative_net.keras"
    plugin.save(model_path)
    
    loaded_plugin = Plugin()
    loaded_plugin.load(model_path)
    
    x_prime_t_loaded = loaded_plugin.decode([dummy_z_t, dummy_h_t, dummy_conditions_t])
    print(f"Test loaded decode output shape: x_prime_t: {x_prime_t_loaded.shape}")
    np.testing.assert_array_almost_equal(x_prime_t, x_prime_t_loaded)
    print("\nSave, load, and re-decode test successful.")
    
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
