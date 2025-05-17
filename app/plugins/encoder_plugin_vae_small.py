import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2

class Plugin:
    """
    Plugin to define and manage a per-step inference network (encoder-like component)
    for a sequential, conditional generative model (e.g., CVAE for time series).
    This network takes a window of current inputs (x_window_t), a recurrent context (h_{t-1}),
    and other conditions (conditions_t) to output parameters for the latent variable z_t.
    """

    plugin_params = {
        "conv_activation": "relu",
        "dense_activation": "relu",
        'learning_rate': 0.0001, # Example, part of the larger model's optimizer
        "l2_reg": 1e-5,
        "dropout_rate": 0.2,
        "conv_layers_config": [ # Example: list of dicts for Conv1D layers
            {"filters": 32, "kernel_size": 5, "strides": 1, "padding": "causal"},
            {"filters": 64, "kernel_size": 3, "strides": 1, "padding": "causal"}
        ],
        "use_batch_norm_conv": True,
        "use_max_pooling": True,
        "pool_size": 2,
        "dense_layer_sizes_after_concat": [128, 64], # Dense layers after concatenating conv output, h_prev, conditions
        # Parameters to be configured:
        "window_size": None, # Number of time steps in the input window
        "input_features_per_step": None, # Dimensionality of features at each step in the input window
        "rnn_hidden_dim": None, # Dimensionality of h_{t-1}
        "conditioning_dim": None, # Dimensionality of other conditions (e.g., previous step's 6 target features)
        "latent_dim": None, # Output latent dimension for z_t
    }

    plugin_debug_vars = [
        'window_size', 'input_features_per_step', 'rnn_hidden_dim', 'conditioning_dim', 'latent_dim',
        'conv_layers_config', 'dense_layer_sizes_after_concat', 'conv_activation', 'dense_activation', 
        'l2_reg', 'dropout_rate', 'use_batch_norm_conv', 'use_max_pooling'
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.inference_network_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params: # Only update known params or allow adding new ones
                self.params[key] = value
            elif key in ["conv_layers_config", "dense_layer_sizes_after_concat"]: # Deep copy for mutable defaults
                 self.params[key] = [item.copy() if isinstance(item, dict) else item for item in value]


    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_model_architecture(self, window_size: int, input_features_per_step: int, 
                                     rnn_hidden_dim: int, conditioning_dim: int, latent_dim: int, 
                                     config: dict = None):
        """
        Configures the per-step inference network with Conv1D layers for windowed input.

        Args:
            window_size (int): Number of time steps in the input window.
            input_features_per_step (int): Dimensionality of features at each step in the input window.
            rnn_hidden_dim (int): Dimensionality of the RNN's previous hidden state h_{t-1}.
            conditioning_dim (int): Dimensionality of other concatenated conditioning variables for step t.
            latent_dim (int): Desired dimensionality for the latent variable z_t.
            config (dict, optional): External configuration dictionary to override plugin_params.
        """
        if config is None:
            config = {}

        # Update internal params with provided arguments and external config
        self.params['window_size'] = window_size
        self.params['input_features_per_step'] = input_features_per_step
        self.params['rnn_hidden_dim'] = rnn_hidden_dim
        self.params['conditioning_dim'] = conditioning_dim
        self.params['latent_dim'] = latent_dim

        # Get architectural parameters, prioritizing external config, then self.params
        conv_activation = config.get("conv_activation", self.params.get("conv_activation", "relu"))
        dense_activation = config.get("dense_activation", self.params.get("dense_activation", "relu"))
        l2_reg_val = config.get("l2_reg", self.params.get("l2_reg", 1e-5))
        dropout_rate = config.get("dropout_rate", self.params.get("dropout_rate", 0.2))
        
        conv_layers_config = config.get("conv_layers_config", self.params.get("conv_layers_config"))
        use_batch_norm_conv = config.get("use_batch_norm_conv", self.params.get("use_batch_norm_conv", True))
        use_max_pooling = config.get("use_max_pooling", self.params.get("use_max_pooling", True))
        pool_size = config.get("pool_size", self.params.get("pool_size", 2))
        
        dense_layer_sizes_after_concat = config.get("dense_layer_sizes_after_concat", self.params.get("dense_layer_sizes_after_concat"))
        
        print(f"[DEBUG EncoderPlugin] Configuring with: window_size={window_size}, input_features={input_features_per_step}, "
              f"h_dim={rnn_hidden_dim}, cond_dim={conditioning_dim}, z_dim={latent_dim}")
        print(f"[DEBUG EncoderPlugin] Conv Layers: {conv_layers_config}, Dense Layers (post-concat): {dense_layer_sizes_after_concat}")

        # --- Define Model ---
        input_x_window = Input(shape=(window_size, input_features_per_step), name="input_x_window")
        input_h_prev = Input(shape=(rnn_hidden_dim,), name="input_h_prev")
        input_conditions_t = Input(shape=(conditioning_dim,), name="input_conditions_t")

        # Convolutional part for processing the window
        x_conv = input_x_window
        for i, conv_config in enumerate(conv_layers_config):
            x_conv = Conv1D(
                filters=conv_config["filters"],
                kernel_size=conv_config["kernel_size"],
                strides=conv_config.get("strides", 1),
                padding=conv_config.get("padding", "causal"),
                kernel_regularizer=l2(l2_reg_val),
                name=f"conv1d_layer_{i+1}"
            )(x_conv)
            if use_batch_norm_conv:
                x_conv = BatchNormalization(name=f"bn_conv_{i+1}")(x_conv)
            if conv_activation == 'leakyrelu':
                x_conv = LeakyReLU(name=f"leakyrelu_conv_{i+1}")(x_conv)
            else:
                x_conv = Dense(0, activation=conv_activation, name=f"{conv_activation}_conv_{i+1}")(x_conv) # Placeholder for activation if not LeakyReLU
                # A bit of a hack for general activation, Keras Conv1D takes activation directly
                # For simplicity, let's assume conv_activation is directly usable or handle specific cases
                # Reverting to direct activation in Conv1D if simple
                # x_conv = Conv1D(..., activation=conv_activation)(x_conv) # Simpler if activation is standard

            if use_max_pooling and conv_config.get("pool_after", True): # Optional pooling after each conv
                 x_conv = MaxPooling1D(pool_size=pool_size, name=f"maxpool_{i+1}")(x_conv)
            if dropout_rate > 0:
                x_conv = Dropout(dropout_rate, name=f"dropout_conv_{i+1}")(x_conv)
        
        # Flatten the output of Conv layers
        flattened_conv_output = Flatten(name="flatten_conv_output")(x_conv)

        # Concatenate flattened conv output with h_prev and conditions_t
        concatenated_features = Concatenate(name="concat_features")([flattened_conv_output, input_h_prev, input_conditions_t])

        # Dense layers after concatenation
        x_dense = concatenated_features
        for i, units in enumerate(dense_layer_sizes_after_concat):
            x_dense = Dense(
                units,
                kernel_regularizer=l2(l2_reg_val),
                name=f"dense_layer_post_concat_{i+1}"
            )(x_dense)
            # Batch Norm and Activation for Dense layers
            x_dense = BatchNormalization(name=f"bn_dense_{i+1}")(x_dense)
            if dense_activation == 'leakyrelu':
                 x_dense = LeakyReLU(name=f"leakyrelu_dense_{i+1}")(x_dense)
            else:
                 x_dense = Dense(0, activation=dense_activation, name=f"{dense_activation}_dense_{i+1}")(x_dense) # Placeholder for activation
                 # x_dense = Dense(units, activation=dense_activation, ...)(x_dense) # Simpler

            if dropout_rate > 0:
                x_dense = Dropout(dropout_rate, name=f"dropout_dense_{i+1}")(x_dense)


        # Output layers for z_mean and z_log_var
        z_mean = Dense(latent_dim, name='z_mean_t', kernel_regularizer=l2(l2_reg_val))(x_dense)
        z_log_var = Dense(latent_dim, name='z_log_var_t', kernel_regularizer=l2(l2_reg_val))(x_dense)

        self.inference_network_model = Model(
            inputs=[input_x_window, input_h_prev, input_conditions_t],
            outputs=[z_mean, z_log_var],
            name="conv1d_cvae_encoder"
        )
        
        print(f"[DEBUG EncoderPlugin] Model built.")
        self.inference_network_model.summary()

    def train(self, *args, **kwargs):
        print("WARNING: EncoderPlugin.train() called. This component is typically trained as part of a larger CVAE model.")
        pass

    def encode(self, per_step_inputs: list):
        """
        Processes the inputs (window, h_prev, conditions) to produce z_mean and z_log_var.

        Args:
            per_step_inputs (list): A list containing [x_window_batch, h_prev_batch, conditions_t_batch].
                                   Shapes:
                                   - x_window_batch: (batch_size, window_size, input_features_per_step)
                                   - h_prev_batch: (batch_size, rnn_hidden_dim)
                                   - conditions_t_batch: (batch_size, conditioning_dim)
        Returns:
            tuple: (z_mean_batch, z_log_var_batch)
        """
        if not self.inference_network_model:
            raise ValueError("Encoder model is not configured or loaded.")
        if not isinstance(per_step_inputs, list) or len(per_step_inputs) != 3:
            raise ValueError("per_step_inputs must be a list of three numpy arrays: [x_window_batch, h_prev_batch, conditions_t_batch]")
        
        # print(f"[EncoderPlugin] Encoding with input shapes: {[data.shape for data in per_step_inputs]}")
        z_mean_batch, z_log_var_batch = self.inference_network_model.predict(per_step_inputs, verbose=0)
        # print(f"[EncoderPlugin] Produced z_mean shape: {z_mean_batch.shape}, z_log_var shape: {z_log_var_batch.shape}")
        return z_mean_batch, z_log_var_batch

    def save(self, file_path):
        if self.inference_network_model:
            save_model(self.inference_network_model, file_path)
            print(f"Encoder model saved to {file_path}")
        else:
            print("Encoder model not available to save.")

    def load(self, file_path, compile_model=False):
        self.inference_network_model = load_model(file_path, compile=compile_model)
        print(f"Encoder model loaded from {file_path}")
        
        try:
            input_layers = self.inference_network_model.inputs
            self.params['window_size'] = input_layers[0].shape[1]
            self.params['input_features_per_step'] = input_layers[0].shape[2]
            self.params['rnn_hidden_dim'] = input_layers[1].shape[-1]
            self.params['conditioning_dim'] = input_layers[2].shape[-1]
            
            output_layers = self.inference_network_model.outputs
            self.params['latent_dim'] = output_layers[0].shape[-1]
            
            print(f"[DEBUG EncoderPlugin Load] Reconstructed params from model: "
                  f"window_size={self.params['window_size']}, input_features={self.params['input_features_per_step']}, "
                  f"h_dim={self.params['rnn_hidden_dim']}, cond_dim={self.params['conditioning_dim']}, "
                  f"z_dim={self.params['latent_dim']}")
        except Exception as e:
            print(f"[WARNING EncoderPlugin Load] Could not fully reconstruct params from loaded model. Error: {e}.")

# Example usage
if __name__ == "__main__":
    plugin = Plugin()
    
    _window_size = 20
    _input_features = 50 # e.g., O,H,L,C,Vol + many TIs + seasonal
    _h_dim = 64
    _cond_dim = 6 # e.g., previous step's 6 target features (O,L,H,C,BC-BO,BH-BL)
    _latent_dim = 32
    
    # Example config override for testing
    test_config = {
        "conv_layers_config": [
            {"filters": 16, "kernel_size": 5, "strides": 1, "padding": "causal", "pool_after": True},
            {"filters": 32, "kernel_size": 3, "strides": 1, "padding": "causal", "pool_after": False}
        ],
        "dense_layer_sizes_after_concat": [64],
        "dropout_rate": 0.1,
        "conv_activation": "relu", # Changed to relu for direct use in Conv1D
        "dense_activation": "relu" # Changed to relu for direct use in Dense
    }

    plugin.configure_model_architecture(
        window_size=_window_size,
        input_features_per_step=_input_features,
        rnn_hidden_dim=_h_dim,
        conditioning_dim=_cond_dim,
        latent_dim=_latent_dim,
        config=test_config
    )
    
    batch_size = 4
    dummy_x_window = np.random.rand(batch_size, _window_size, _input_features).astype(np.float32)
    dummy_h_prev = np.random.rand(batch_size, _h_dim).astype(np.float32)
    dummy_conditions_t = np.random.rand(batch_size, _cond_dim).astype(np.float32)
    
    z_mean, z_log_var = plugin.encode([dummy_x_window, dummy_h_prev, dummy_conditions_t])
    print(f"\nTest encode output shapes: z_mean: {z_mean.shape}, z_log_var: {z_log_var.shape}")

    model_path = "temp_conv1d_cvae_encoder.keras"
    plugin.save(model_path)
    loaded_plugin = Plugin()
    loaded_plugin.load(model_path)
    z_mean_loaded, z_log_var_loaded = loaded_plugin.encode([dummy_x_window, dummy_h_prev, dummy_conditions_t])
    print(f"Test loaded encode output shapes: z_mean: {z_mean_loaded.shape}, z_log_var: {z_log_var_loaded.shape}")
    np.testing.assert_array_almost_equal(z_mean, z_mean_loaded)
    print("\nSave, load, and re-encode test successful.")
    
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
