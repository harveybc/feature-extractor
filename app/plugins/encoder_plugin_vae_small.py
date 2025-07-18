import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Conv1D, Flatten, LSTM, Bidirectional, RepeatVector, TimeDistributed, MultiHeadAttention
from tensorflow.keras.layers import LeakyReLU, LayerNormalization, Add, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K





def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # Apply sin to even indices; cos to odd indices
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]  # Shape: (1, position, d_model)
    return tf.cast(pos_encoding, dtype=tf.float32)
    
class Plugin:
    """
    Plugin to define and manage a per-step inference network (encoder-like component)
    for a sequential, conditional generative model (e.g., CVAE for time series).
    This network takes a window of current inputs (x_window_t), a recurrent context (h_{t-1}),
    and other conditions (conditions_t) to output parameters for the latent variable z_t.
    Uses a configurable Conv1D -> BiLSTM architecture.
    """

    plugin_params = {
        "conv_activation": "relu",
        'learning_rate': 0.0001, 
        "l2_reg": 1e-5,
        "initial_conv_filters": 128, 
        "conv_kernel_size": 5,      
        "conv_padding": "same",     
        "num_conv_layers": 4,           # Total number of Conv1D layers
        "num_strided_conv_layers": 2,   # Number of initial Conv1D layers that will have strides=2
        "min_conv_filters": 16,         # Minimum number of filters for a conv layer
        "lstm_units": 64,               # Configurable LSTM units for each direction of BiLSTM
        # Parameters to be configured by AutoencoderManager:
        "window_size": None, 
        "input_features_per_step": None, 
        "rnn_hidden_dim": None, 
        "conditioning_dim": None, 
        "latent_dim": None, 
    }

    plugin_debug_vars = [
        'window_size', 'input_features_per_step', 'rnn_hidden_dim', 'conditioning_dim', 'latent_dim',
        'initial_conv_filters', 'conv_kernel_size', 'conv_padding', 'conv_activation', 
        'num_conv_layers', 'num_strided_conv_layers', 'min_conv_filters', 'lstm_units', 'l2_reg'
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

    def add_debug_info(self, debug_info): # Not strictly needed if not called by manager
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def configure_model_architecture(self, window_size: int, input_features_per_step: int, 
                                     rnn_hidden_dim: int, conditioning_dim: int, latent_dim: int, 
                                     config= None):
        if config is None:
            config = {}

        self.params['window_size'] = window_size
        self.params['input_features_per_step'] = input_features_per_step
        self.params['rnn_hidden_dim'] = rnn_hidden_dim
        self.params['conditioning_dim'] = conditioning_dim
        self.params['latent_dim'] = latent_dim

        conv_activation_str = config.get("conv_activation", self.params.get("conv_activation", "relu"))
        l2_reg_val = config.get("l2_reg", self.params.get("l2_reg", 1e-5))
        initial_conv_filters = config.get("initial_conv_filters", self.params.get("initial_conv_filters", 128))
        conv_kernel_size = config.get("conv_kernel_size", self.params.get("conv_kernel_size", 5))
        conv_padding = config.get("conv_padding", self.params.get("conv_padding", "same"))
        num_conv_layers_cfg = config.get("num_conv_layers", self.params.get("num_conv_layers", 4))
        num_strided_conv_layers_cfg = config.get("num_strided_conv_layers", self.params.get("num_strided_conv_layers", 2))
        min_conv_filters_cfg = config.get("min_conv_filters", self.params.get("min_conv_filters", 16))
        lstm_units_cfg = config.get("lstm_units", self.params.get("lstm_units", 64))
        output_activation_name = config.get("output_activation", self.params.get("output_activation", "linear"))
        
        print(f"[DEBUG EncoderPlugin] Configuring with: window_size={window_size}, input_features={input_features_per_step}, "
              f"h_dim={rnn_hidden_dim}, cond_dim={conditioning_dim}, z_dim={latent_dim}")
        print(f"[DEBUG EncoderPlugin] Conv Params: initial_filters={initial_conv_filters}, kernel_size={conv_kernel_size}, "
              f"padding='{conv_padding}', activation='{conv_activation_str}', l2_reg={l2_reg_val}")
        print(f"[DEBUG EncoderPlugin] Layers: num_conv={num_conv_layers_cfg}, num_strided={num_strided_conv_layers_cfg}, min_filters={min_conv_filters_cfg}")
        print(f"[DEBUG EncoderPlugin] LSTM Units (per direction): {lstm_units_cfg}")

        input_x_window = Input(shape=(window_size, input_features_per_step), name="input_x_window")
        input_h_prev = Input(shape=(rnn_hidden_dim,), name="input_h_prev")
        input_conditions_t = Input(shape=(conditioning_dim,), name="input_conditions_t")

        # --- MODIFIED: Concatenate context and conditions with input_x_window ---
        # Expand h_prev and conditions_t to match the window_size dimension
        h_prev_expanded = RepeatVector(window_size, name="h_prev_repeated")(input_h_prev)
        conditions_t_expanded = RepeatVector(window_size, name="conditions_t_repeated")(input_conditions_t)

        # Concatenate along the feature axis
        concatenated_input_for_conv = Concatenate(axis=-1, name="concat_input_with_context_conditions")(
            [input_x_window, h_prev_expanded, conditions_t_expanded]
        )
        
        # --- NEW: Early MultiHeadAttention ---
        # Calculate key_dim for MHA based on the feature dimension of concatenated_input_for_conv
        # These params are set at the start of this function.
        d_concat_features = self.params['input_features_per_step'] + self.params['rnn_hidden_dim'] + self.params['conditioning_dim']
        #early_attn_key_dim = max(1, d_concat_features // 4) # Assuming num_heads=2, this aims to halve feature dim


        x=concatenated_input_for_conv
        # Add positional encoding to capture temporal order
        # get static shape tuple via Keras backend
        last_layer_shape = K.int_shape(x)
        feature_dim = last_layer_shape[-1]
        # get the sequence length from the last layer shape
        seq_length = last_layer_shape[1]
        pos_enc = positional_encoding(seq_length, feature_dim)
        x = x + pos_enc

        # --- Self-Attention Block 1 ---
        num_attention_heads = 2
        # get the last layer shape from the merged tensor
        last_layer_shape = K.int_shape(x)
        # get the feature dimension from the last layer shape as the last component of the shape tuple
        feature_dim = last_layer_shape[-1]
        # define key dimension for attention    
        attention_key_dim = feature_dim//num_attention_heads
        # Apply MultiHeadAttention
        attention_output = MultiHeadAttention(
            num_heads=num_attention_heads, # Assumed to be defined
            key_dim=attention_key_dim,      # Assumed to be defined
            kernel_regularizer=l2(l2_reg_val),
            name=f"multihead_attention_1"
        )(query=x, value=x, key=x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        
        x_conv = x # Input to Conv1D stack is now the output of early attention
        # --- END NEW Early MultiHeadAttention ---
        
        current_layer_filters = initial_conv_filters
        strides_for_layer = 2  # FIXED: Always use stride=2 to halve temporal dimension

        for i in range(num_conv_layers_cfg):
    
            x_conv = Conv1D(
                filters=current_layer_filters,
                kernel_size=3,  # FIXED: Use kernel_size=3 as you specified
                strides=strides_for_layer,
                padding=conv_padding,
                activation=None,  # Use LeakyReLU separately
                kernel_regularizer=l2(l2_reg_val),
                kernel_initializer=HeNormal(),
                name=f"conv1d_layer_{i+1}"
            )(x_conv)
            x_conv = LeakyReLU(alpha=0.2, name=f"conv1d_layer_{i+1}_leaky")(x_conv)
            current_layer_filters = max(min_conv_filters_cfg, current_layer_filters // 2)  # Halve filters each layer, but not below min

        #batch normalization
        x_conv = BatchNormalization(name="conv1d_batch_norm")(x_conv) 

        bilstm_output = Bidirectional(
            LSTM(units=lstm_units_cfg,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 return_sequences=True),
                 #kernel_regularizer=l2(l2_reg_val),
                 #recurrent_regularizer=l2(l2_reg_val)),
            name="bilstm_layer"
        )(x_conv)  # shape (batch, window_size, 2*lstm_units_cfg)

        # self-attention over time - THIS IS NOW DONE EARLIER
        # attn = MultiHeadAttention(
        #     num_heads=2,
        #     key_dim=2*lstm_units_cfg // 4,
        #     name="self_attention"
        # )(bilstm_output, bilstm_output)  # (batch, window_size, 2*lstm_units_cfg)

        # project to latent sequence - TAKES INPUT FROM BILSTM DIRECTLY NOW
        z_mean_seq = TimeDistributed(
            #Dense(latent_dim, kernel_regularizer=l2(l2_reg_val)),
            Dense(latent_dim),
            name="z_mean_seq"
        )(bilstm_output) # CHANGED: Input is bilstm_output
        z_log_var_seq = TimeDistributed(
            #Dense(latent_dim, kernel_regularizer=l2(l2_reg_val)),
            Dense(latent_dim),
            name="z_log_var_seq"
        )(bilstm_output) # CHANGED: Input is bilstm_output

        # final 3D outputs
        z_mean, z_log_var = z_mean_seq, z_log_var_seq

        self.inference_network_model = Model(
            inputs=[input_x_window, input_h_prev, input_conditions_t],
            outputs=[z_mean, z_log_var],
            name="configurable_conv_bilstm_cvae_encoder"
        )
        
        print(f"[DEBUG EncoderPlugin] Model built.")
        self.inference_network_model.summary(line_length=120)

    def train(self, *args, **kwargs):
        print("WARNING: EncoderPlugin.train() called. This component is typically trained as part of a larger CVAE model.")
        pass

    def encode(self, per_step_inputs: list):
        if not self.inference_network_model:
            raise ValueError("Encoder model is not configured or loaded.")
        if not isinstance(per_step_inputs, list) or len(per_step_inputs) != 3:
            raise ValueError("per_step_inputs must be a list of three numpy arrays: [x_window_batch, h_prev_batch, conditions_t_batch]")
        
        z_mean_batch, z_log_var_batch = self.inference_network_model.predict(per_step_inputs, verbose=0)
        return z_mean_batch, z_log_var_batch

    def save(self, file_path):
        if self.inference_network_model:
            # Use native Keras format (.keras) to avoid legacy HDF5 warnings
            if file_path.endswith('.h5'):
                tf.print(f"Warning: Converting legacy .h5 path to modern .keras format")
                file_path = file_path.replace('.h5', '.keras')
            self.inference_network_model.save(file_path, save_format='keras')
            print(f"Encoder model saved to {file_path}")
        else:
            print("Encoder model not available to save.")

    def load(self, file_path, compile_model=False):
        # Handle both legacy .h5 and modern .keras formats
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
            
            # Note: Reconstructing num_conv_layers, num_strided_conv_layers, lstm_units, etc.
            # from a loaded model purely by inspecting layers can be complex and error-prone.
            # It's better if these are known or if the model is always built with current params.
            # For now, we only reconstruct the I/O dimensions.
            print(f"[DEBUG EncoderPlugin Load] Reconstructed I/O params from model: "
                  f"window_size={self.params['window_size']}, input_features={self.params['input_features_per_step']}, "
                  f"h_dim={self.params['rnn_hidden_dim']}, cond_dim={self.params['conditioning_dim']}, "
                  f"z_dim={self.params['latent_dim']}")
        except Exception as e:
            print(f"[WARNING EncoderPlugin Load] Could not fully reconstruct params from loaded model. Error: {e}.")

if __name__ == "__main__":
    plugin = Plugin()
    
    _window_size = 288 
    _input_features = 54 
    _h_dim = 64
    _cond_dim = 10 # Updated to match typical conditioning_dim
    _latent_dim = 32
    
    test_config = {
        "initial_conv_filters": 128,
        "conv_kernel_size": 5,
        "num_conv_layers": 4,
        "num_strided_conv_layers": 2, # Downsample by 2*2 = 4 times
        "min_conv_filters": 32,
        "lstm_units": 64, # Units for each direction of BiLSTM
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

    model_path = "temp_configurable_encoder.keras"
    plugin.save(model_path)
    loaded_plugin = Plugin()
    loaded_plugin.load(model_path) # Loads the model structure
    # For a true test of loaded config, you might re-configure then load weights,
    # or ensure loaded_plugin.params are updated if possible.
    # Here, loaded_plugin will use its default params unless configure_model_architecture is called again.
    # The loaded model itself is correct, however.
    
    print("\nSimulating encode with loaded model (architecture is fixed as per saved file):")
    z_mean_loaded, z_log_var_loaded = loaded_plugin.encode([dummy_x_window, dummy_h_prev, dummy_conditions_t])
    print(f"Test loaded encode output shapes: z_mean: {z_mean_loaded.shape}, z_log_var: {z_log_var_loaded.shape}")
    np.testing.assert_array_almost_equal(z_mean, z_mean_loaded)
    print("\nSave, load, and re-encode test successful.")
    
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
