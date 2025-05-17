import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Concatenate, Conv1D, Flatten, LSTM, Bidirectional # Removed MaxPooling, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras.regularizers import l2

class Plugin:
    """
    Plugin to define and manage a per-step inference network (encoder-like component)
    for a sequential, conditional generative model (e.g., CVAE for time series).
    This network takes a window of current inputs (x_window_t), a recurrent context (h_{t-1}),
    and other conditions (conditions_t) to output parameters for the latent variable z_t.
    Uses a simplified Conv1D -> BiLSTM architecture.
    """

    plugin_params = {
        "conv_activation": "relu",
        'learning_rate': 0.0001, 
        "l2_reg": 1e-5,
        # Parameters for the 4 Conv1D layers
        "initial_conv_filters": 128, # Filters for the first Conv1D layer, e.g., 128 -> 64 -> 32 -> 16
        "conv_kernel_size": 5,      # Kernel size for all Conv1D layers
        "conv_padding": "same",     # Padding for Conv1D layers
        # Parameters to be configured:
        "window_size": None, 
        "input_features_per_step": None, 
        "rnn_hidden_dim": None, 
        "conditioning_dim": None, 
        "latent_dim": None, 
    }

    plugin_debug_vars = [
        'window_size', 'input_features_per_step', 'rnn_hidden_dim', 'conditioning_dim', 'latent_dim',
        'initial_conv_filters', 'conv_kernel_size', 'conv_padding', 'conv_activation', 
        'l2_reg'
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.inference_network_model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params: 
                self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_model_architecture(self, window_size: int, input_features_per_step: int, 
                                     rnn_hidden_dim: int, conditioning_dim: int, latent_dim: int, 
                                     config: dict = None):
        """
        Configures the per-step inference network with a fixed 4-layer Conv1D architecture
        followed by a Bidirectional LSTM.

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
        conv_activation_str = config.get("conv_activation", self.params.get("conv_activation", "relu"))
        l2_reg_val = config.get("l2_reg", self.params.get("l2_reg", 1e-5))
        initial_conv_filters = config.get("initial_conv_filters", self.params.get("initial_conv_filters", 128))
        conv_kernel_size = config.get("conv_kernel_size", self.params.get("conv_kernel_size", 5))
        conv_padding = config.get("conv_padding", self.params.get("conv_padding", "same"))
        
        print(f"[DEBUG EncoderPlugin] Configuring with: window_size={window_size}, input_features={input_features_per_step}, "
              f"h_dim={rnn_hidden_dim}, cond_dim={conditioning_dim}, z_dim={latent_dim}")
        print(f"[DEBUG EncoderPlugin] Conv Params: initial_filters={initial_conv_filters}, kernel_size={conv_kernel_size}, "
              f"padding='{conv_padding}', activation='{conv_activation_str}', l2_reg={l2_reg_val}")

        # --- Define Model ---
        input_x_window = Input(shape=(window_size, input_features_per_step), name="input_x_window")
        input_h_prev = Input(shape=(rnn_hidden_dim,), name="input_h_prev")
        input_conditions_t = Input(shape=(conditioning_dim,), name="input_conditions_t")

        # Convolutional part for processing the window
        x_conv = input_x_window
        current_filters = initial_conv_filters

        for i in range(4): # 4 Conv1D layers
            x_conv = Conv1D(
                filters=current_filters,
                kernel_size=conv_kernel_size,
                strides=2, # Downsample
                padding=conv_padding,
                activation=conv_activation_str,
                kernel_regularizer=l2(l2_reg_val),
                name=f"conv1d_layer_{i+1}"
            )(x_conv)
            if i < 3: # For the first 3 layers, halve the filters for the next layer
                current_filters = max(1, current_filters // 2) # Ensure filters don't go below 1

        # Bidirectional LSTM layer
        # The number of units for LSTM will be the number of filters in the last Conv1D layer
        lstm_units = current_filters 
        print(f"[DEBUG EncoderPlugin] LSTM units (derived from last Conv1D filters): {lstm_units}")
        bilstm_output = Bidirectional(
            LSTM(units=lstm_units, 
                 activation='tanh', # Common activation for LSTM
                 recurrent_activation='sigmoid', # Common activation for LSTM
                 kernel_regularizer=l2(l2_reg_val),
                 recurrent_regularizer=l2(l2_reg_val),
                 return_sequences=False), # Output only the final state
            name="bilstm_layer"
        )(x_conv) # Input to BiLSTM is the output of the last Conv1D layer

        # Concatenate BiLSTM output with h_prev and conditions_t
        # Note: BiLSTM output is already flattened because return_sequences=False
        concatenated_features = Concatenate(name="concat_features")([bilstm_output, input_h_prev, input_conditions_t])

        # Output layers for z_mean and z_log_var, fed directly from concatenated_features
        z_mean = Dense(latent_dim, name='z_mean_t', kernel_regularizer=l2(l2_reg_val))(concatenated_features)
        z_log_var = Dense(latent_dim, name='z_log_var_t', kernel_regularizer=l2(l2_reg_val))(concatenated_features)

        self.inference_network_model = Model(
            inputs=[input_x_window, input_h_prev, input_conditions_t],
            outputs=[z_mean, z_log_var],
            name="simplified_conv_bilstm_cvae_encoder"
        )
        
        print(f"[DEBUG EncoderPlugin] Model built.")
        self.inference_network_model.summary(line_length=120)

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
    
    _window_size = 288 # Example: 1 day of 5-min intervals
    _input_features = 54 
    _h_dim = 64
    _cond_dim = 6 
    _latent_dim = 32
    
    # Example config override for testing the simplified architecture
    test_config = {
        "initial_conv_filters": 128, # e.g., 128 -> 64 -> 32 -> 16
        "conv_kernel_size": 5,
        "conv_padding": "same",
        "conv_activation": "relu",
        "l2_reg": 1e-5
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
