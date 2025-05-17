import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Concatenate # Changed imports
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2
# Removed unused imports like Conv1D, MaxPooling1D, Flatten, etc.
# from keras.callbacks import EarlyStopping # Callbacks are usually for training the whole model
# from keras.layers import BatchNormalization, LeakyReLU, Reshape
# from tensorflow.keras.losses import Huber
# from tensorflow.keras.layers import Bidirectional, LSTM


class Plugin:
    """
    Plugin to define and manage a per-step inference network (encoder-like component)
    for a sequential, conditional generative model (e.g., VRNN).
    This network takes current inputs (x_t, h_{t-1}, conditions_t) and outputs
    parameters for the latent variable z_t (z_mean_t, z_log_var_t).
    """

    plugin_params = {
        "activation": "relu", # Common for intermediate dense layers
        'learning_rate': 0.0001, # Example, will be part of the larger model's optimizer
        "l2_reg": 1e-5,
        "dense_layer_sizes": [128, 64], # Example intermediate layer sizes
        # Parameters to be configured:
        "x_feature_dim": None, # Dimensionality of x_t (e.g., 6 base features)
        "rnn_hidden_dim": None, # Dimensionality of h_{t-1}
        "conditioning_dim": None, # Dimensionality of other conditions (seasonal, current fundamentals)
        "latent_dim": None, # Output latent dimension for z_t
    }

    plugin_debug_vars = [
        'x_feature_dim', 'rnn_hidden_dim', 'conditioning_dim', 'latent_dim',
        'dense_layer_sizes', 'activation', 'l2_reg'
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.inference_network_model = None # Renamed from encoder_model
        # self.shape_before_flatten_for_decoder is no longer relevant for this type of model

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value # Allow setting any, or check against self.params.keys()

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_model_architecture(self, x_feature_dim: int, rnn_hidden_dim: int, conditioning_dim: int, latent_dim: int, config: dict = None):
        """
        Configures the per-step inference network.

        Args:
            x_feature_dim (int): Dimensionality of the observable x_t at the current step.
            rnn_hidden_dim (int): Dimensionality of the RNN's previous hidden state h_{t-1}.
            conditioning_dim (int): Dimensionality of other concatenated conditioning variables for step t.
            latent_dim (int): Desired dimensionality for the latent variable z_t.
            config (dict, optional): External configuration dictionary to override plugin_params.
        """
        if config is None:
            config = {}

        self.params['x_feature_dim'] = x_feature_dim
        self.params['rnn_hidden_dim'] = rnn_hidden_dim
        self.params['conditioning_dim'] = conditioning_dim
        self.params['latent_dim'] = latent_dim

        # Get parameters, prioritizing external config, then self.params (which has defaults)
        activation = config.get("activation", self.params.get("activation", "relu"))
        l2_reg_val = config.get("l2_reg", self.params.get("l2_reg", 1e-5))
        dense_layer_sizes = config.get("dense_layer_sizes", self.params.get("dense_layer_sizes", [128, 64]))
        
        print(f"[DEBUG PerStepInferenceNetwork] Configuring with: x_dim={x_feature_dim}, h_dim={rnn_hidden_dim}, cond_dim={conditioning_dim}, z_dim={latent_dim}")
        print(f"[DEBUG PerStepInferenceNetwork] Dense layers: {dense_layer_sizes}, Activation: {activation}, L2: {l2_reg_val}")

        # Define inputs for a single time step
        # Note: x_t_input is primarily used during training when actual x_t is available.
        # For generation, if x_t is what's being generated, it might not be an input here,
        # or this network is part of a larger structure. Assuming VRNN-like training for now.
        input_x_t = Input(shape=(x_feature_dim,), name="input_x_t")
        input_h_prev = Input(shape=(rnn_hidden_dim,), name="input_h_prev")
        input_conditions_t = Input(shape=(conditioning_dim,), name="input_conditions_t")

        # Concatenate all inputs
        concatenated_inputs = Concatenate(name="concat_inputs")([input_x_t, input_h_prev, input_conditions_t])

        # Intermediate Dense layers
        x = concatenated_inputs
        for i, units in enumerate(dense_layer_sizes):
            x = Dense(
                units,
                activation=activation,
                kernel_regularizer=l2(l2_reg_val),
                name=f"dense_layer_{i+1}"
            )(x)

        # Output layers for z_mean and z_log_var
        z_mean = Dense(latent_dim, name='z_mean_t', kernel_regularizer=l2(l2_reg_val))(x)
        z_log_var = Dense(latent_dim, name='z_log_var_t', kernel_regularizer=l2(l2_reg_val))(x)

        self.inference_network_model = Model(
            inputs=[input_x_t, input_h_prev, input_conditions_t],
            outputs=[z_mean, z_log_var],
            name="per_step_inference_network"
        )
        
        print(f"[DEBUG PerStepInferenceNetwork] Model built. Inputs: {[input_x_t, input_h_prev, input_conditions_t]}, Outputs: {[z_mean, z_log_var]}")
        self.inference_network_model.summary() # Print model summary

    def train(self, *args, **kwargs):
        # This plugin defines a component. Training is handled by the main sequential model
        # (e.g., the VRNN model in synthetic-datagen's OptimizerPlugin).
        # This method could be used for pre-training this component if desired,
        # but that would require a specific setup.
        print("WARNING: PerStepInferenceNetworkPlugin.train() called. This component is typically trained as part of a larger sequential model.")
        pass

    def encode(self, per_step_inputs: list):
        """
        Processes the inputs for a single time step (or a batch of time steps)
        to produce z_mean and z_log_var.

        Args:
            per_step_inputs (list): A list containing [x_t_batch, h_prev_batch, conditions_t_batch].
                                   Shapes:
                                   - x_t_batch: (batch_size, x_feature_dim)
                                   - h_prev_batch: (batch_size, rnn_hidden_dim)
                                   - conditions_t_batch: (batch_size, conditioning_dim)
        Returns:
            tuple: (z_mean_batch, z_log_var_batch)
        """
        if not self.inference_network_model:
            raise ValueError("Per-step inference network model is not configured or loaded.")
        if not isinstance(per_step_inputs, list) or len(per_step_inputs) != 3:
            raise ValueError("per_step_inputs must be a list of three numpy arrays: [x_t_batch, h_prev_batch, conditions_t_batch]")
        
        print(f"[PerStepInferenceNetwork] Encoding with input shapes: {[data.shape for data in per_step_inputs]}")
        z_mean_batch, z_log_var_batch = self.inference_network_model.predict(per_step_inputs)
        print(f"[PerStepInferenceNetwork] Produced z_mean shape: {z_mean_batch.shape}, z_log_var shape: {z_log_var_batch.shape}")
        return z_mean_batch, z_log_var_batch

    def save(self, file_path):
        if self.inference_network_model:
            # Save the architectural configuration (self.params) alongside the model weights if needed,
            # or ensure the loading mechanism can reconstruct it.
            # For simplicity, just saving the Keras model here.
            save_model(self.inference_network_model, file_path)
            print(f"Per-step inference network model saved to {file_path}")
        else:
            print("Per-step inference network model not available to save.")

    def load(self, file_path, compile_model=False): # Changed compile to compile_model for clarity
        # compile=False is typical if this model is a sub-component or only used for predict.
        self.inference_network_model = load_model(file_path, compile=compile_model)
        print(f"Per-step inference network model loaded from {file_path}")
        
        # Attempt to re-populate params from the loaded model's structure
        try:
            input_layers = self.inference_network_model.inputs
            self.params['x_feature_dim'] = input_layers[0].shape[-1]
            self.params['rnn_hidden_dim'] = input_layers[1].shape[-1]
            self.params['conditioning_dim'] = input_layers[2].shape[-1]
            
            output_layers = self.inference_network_model.outputs
            self.params['latent_dim'] = output_layers[0].shape[-1]
            
            print(f"[DEBUG PerStepInferenceNetwork Load] Reconstructed params from model: "
                  f"x_dim={self.params['x_feature_dim']}, h_dim={self.params['rnn_hidden_dim']}, "
                  f"cond_dim={self.params['conditioning_dim']}, z_dim={self.params['latent_dim']}")
        except Exception as e:
            print(f"[WARNING PerStepInferenceNetwork Load] Could not fully reconstruct params from loaded model structure. Error: {e}. Ensure manual configuration if needed.")

# Example of how this plugin might be used (conceptual, actual use is within synthetic-datagen)
if __name__ == "__main__":
    plugin = Plugin()
    
    # Example dimensions
    _x_dim = 6
    _h_dim = 64
    _cond_dim = 10 # e.g., 6 seasonal + 2 current fundamentals + 2 high-frequency summary
    _latent_dim = 32
    
    plugin.configure_model_architecture(
        x_feature_dim=_x_dim,
        rnn_hidden_dim=_h_dim,
        conditioning_dim=_cond_dim,
        latent_dim=_latent_dim,
        config={"dense_layer_sizes": [100, 50], "activation": "swish"} # Example override
    )
    
    # Create dummy batch data for testing encode
    batch_size = 4
    dummy_x_t = np.random.rand(batch_size, _x_dim).astype(np.float32)
    dummy_h_prev = np.random.rand(batch_size, _h_dim).astype(np.float32)
    dummy_conditions_t = np.random.rand(batch_size, _cond_dim).astype(np.float32)
    
    z_mean, z_log_var = plugin.encode([dummy_x_t, dummy_h_prev, dummy_conditions_t])
    
    print(f"\nTest encode output shapes: z_mean: {z_mean.shape}, z_log_var: {z_log_var.shape}")

    # Test save and load
    model_path = "temp_per_step_inference_net.keras"
    plugin.save(model_path)
    
    loaded_plugin = Plugin()
    loaded_plugin.load(model_path)
    
    # Verify loaded model can predict
    z_mean_loaded, z_log_var_loaded = loaded_plugin.encode([dummy_x_t, dummy_h_prev, dummy_conditions_t])
    print(f"Test loaded encode output shapes: z_mean: {z_mean_loaded.shape}, z_log_var: {z_log_var_loaded.shape}")
    np.testing.assert_array_almost_equal(z_mean, z_mean_loaded) # Check if outputs are the same
    print("\nSave, load, and re-encode test successful.")
    
    # Clean up
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
