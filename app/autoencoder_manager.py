import numpy as np
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import backend as K
from keras.callbacks import LambdaCallback, ReduceLROnPlateau
from keras.layers import Input, Dense, Flatten, Reshape, Dropout, Concatenate # Added Concatenate
from keras.regularizers import l2
from keras.layers import BatchNormalization, LeakyReLU
from keras.layers import MaxPooling1D, UpSampling1D
from keras.layers import Lambda
import keras # Import the top-level keras


#set_global_policy('mixed_float16')

#define tensorflow global variable mmd_total as a float
mmd_total = tf.Variable(0.0, dtype=tf.float32, trainable=False)
# Define new global tf.Variables for additional loss components
huber_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False) # To track Huber loss component
kl_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False) # For KL divergence
skew_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False)
kurtosis_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False)
covariance_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False)

# ============================================================================
# Callback to dynamically adjust mmd_weight so Huber and MMD remain same order
# ============================================================================
class MMDWeightAdjustmentCallback(tf.keras.callbacks.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def on_epoch_end(self, epoch, logs=None):
        # compute current Huber and MMD contributions
        huber_val = logs.get('huber_metric_fn', logs['loss'] - logs.get('kl_metric_fn', 0.0) - logs.get('mmd_metric_fn', 0.0) - logs.get('skew_metric_fn', 0.0) - logs.get('kurtosis_metric_fn', 0.0) - logs.get('covariance_metric_fn', 0.0)) # Use actual metric names
        mmd_val   = logs.get('mmd_metric_fn', 0.0) # Use actual metric name
        ratio     = huber_val / (mmd_val + 1e-12)
        # keep ratio roughly in [0.5, 2.0]
        if ratio > 2.0:
            self.cfg['mmd_weight'] *= 1.1
        elif ratio < 0.5:
            self.cfg['mmd_weight'] *= 0.9
        print(f"[MMDWeightAdjust] Epoch {epoch+1}: huber_val_approx={huber_val:.3f}, mmd_val={mmd_val:.3f}, ratio={ratio:.3f}, new mmd_weight={self.cfg['mmd_weight']:.5f}")

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    """Custom ReduceLROnPlateau callback that prints the patience counter."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.patience_counter = self.wait if self.wait > 0 else 0
        print(f"DEBUG: ReduceLROnPlateau patience counter: {self.patience_counter}")

class EarlyStoppingWithPatienceCounter(EarlyStopping):
    """Custom EarlyStopping callback that prints the patience counter."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.patience_counter = self.wait if self.wait > 0 else 0
        print(f"DEBUG: EarlyStopping patience counter: {self.patience_counter}")


# ---------------------------
# Custom Metrics and Loss Functions
# ---------------------------
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

# Adapt mae_magnitude and r2_metric to take the reconstruction tensor directly
def mae_magnitude(y_true, y_pred_reconstruction): # y_pred_reconstruction is the tensor
    """Compute MAE on the first column (e.g., 'OPEN' price) of the reconstruction."""
    # y_true will also have shape (batch, 6)

    if len(tf.shape(y_true)) != 2 or len(tf.shape(y_pred_reconstruction)) != 2:
        tf.print(f"Warning: mae_magnitude expects 2D y_true (shape {tf.shape(y_true)}) and y_pred_reconstruction (shape {tf.shape(y_pred_reconstruction)}).")
        return 0.0 
    if tf.shape(y_true)[-1] == 0 or tf.shape(y_pred_reconstruction)[-1] == 0:
        return 0.0 

    mag_true = y_true[:, 0:1] 
    mag_pred = y_pred_reconstruction[:, 0:1] 
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

def r2_metric(y_true, y_pred_reconstruction): # y_pred_reconstruction is the tensor
    """Compute R² metric on the first column (e.g., 'OPEN' price) of the reconstruction."""
    # y_true will also have shape (batch, 6)

    if len(tf.shape(y_true)) != 2 or len(tf.shape(y_pred_reconstruction)) != 2:
        tf.print(f"Warning: r2_metric expects 2D y_true (shape {tf.shape(y_true)}) and y_pred_reconstruction (shape {tf.shape(y_pred_reconstruction)}).")
        return 0.0
    if tf.shape(y_true)[-1] == 0 or tf.shape(y_pred_reconstruction)[-1] == 0:
        return 0.0

    mag_true = y_true[:, 0:1] 
    mag_pred = y_pred_reconstruction[:, 0:1] 
    SS_res = tf.reduce_sum(tf.square(mag_true - mag_pred))
    SS_tot = tf.reduce_sum(tf.square(mag_true - tf.reduce_mean(mag_true)))
    return 1 - SS_res/(SS_tot + tf.keras.backend.epsilon())

def compute_mmd(x, y, sigma=1.0, sample_size=32):
    """Compute the Maximum Mean Discrepancy (MMD) between two samples."""
    # Ensure x and y are at least 2D
    if len(tf.shape(x)) < 2: x = tf.reshape(x, [tf.shape(x)[0], -1])
    if len(tf.shape(y)) < 2: y = tf.reshape(y, [tf.shape(y)[0], -1])

    # Ensure x and y have the same number of features if they are 2D
    if len(tf.shape(x)) == 2 and len(tf.shape(y)) == 2 and tf.shape(x)[1] != tf.shape(y)[1]:
        # This case should ideally not happen if y_true and reconstruction have same feature dim
        # Fallback or error, for now, let's print a warning and proceed if possible or return 0
        tf.print(f"Warning: MMD compute feature mismatch x_shape={tf.shape(x)}, y_shape={tf.shape(y)}")
        # Potentially pad or truncate, or return a default value like 0.0
        # For now, let's assume they should match and this is an issue if they don't.
        # If they must match, this could raise an error or return a high MMD.
        # For safety, if features don't match, return 0 to not break training, but log it.
        return 0.0


    min_samples = tf.minimum(tf.shape(x)[0], tf.shape(y)[0])
    if min_samples == 0: return 0.0 # Avoid error if one sample is empty
    
    current_sample_size = tf.minimum(sample_size, min_samples)

    idx_x = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:current_sample_size]
    x_sample = tf.gather(x, idx_x)
    
    idx_y = tf.random.shuffle(tf.range(tf.shape(y)[0]))[:current_sample_size]
    y_sample = tf.gather(y, idx_y)

    def gaussian_kernel(s1, s2, sigma_kernel):
        s1_expanded = tf.expand_dims(s1, 1)
        s2_expanded = tf.expand_dims(s2, 0)
        dist_sq = tf.reduce_sum(tf.square(s1_expanded - s2_expanded), axis=-1)
        return tf.exp(-dist_sq / (2.0 * sigma_kernel ** 2))
        
    K_xx = gaussian_kernel(x_sample, x_sample, sigma)
    K_yy = gaussian_kernel(y_sample, y_sample, sigma)
    K_xy = gaussian_kernel(x_sample, y_sample, sigma)
    return tf.reduce_mean(K_xx) + tf.reduce_mean(K_yy) - 2 * tf.reduce_mean(K_xy)

# ... (calculate_standardized_moment, covariance_loss_calc - assumed to exist and be compatible) ...
# These will operate on y_true and y_pred_list[0] (the reconstruction)
def calculate_standardized_moment(data, order, mean_val=None, std_dev_val=None):
    if tf.size(data) == 0: return tf.constant(0.0, dtype=tf.float32)
    if mean_val is None: mean_val = tf.reduce_mean(data)
    if std_dev_val is None: std_dev_val = tf.math.reduce_std(data)
    if std_dev_val < 1e-6: return tf.constant(0.0, dtype=tf.float32) # Avoid division by zero
    return tf.reduce_mean(((data - mean_val) / std_dev_val) ** order)

def covariance_loss_calc(y_true, y_pred, config):
    # This function needs to be adapted if it's used, to take y_true and y_pred (reconstruction)
    # and calculate covariance loss. For now, assuming it's compatible or will be adapted.
    # Placeholder:
    if not config.get('enable_cov_loss', False):
        covariance_loss_tracker.assign(0.0)
        return 0.0

    y_true_f = tf.cast(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), tf.float32)
    
    # Simplified covariance loss example (difference in covariance matrices' Frobenius norm)
    # This is a placeholder and might need your specific implementation.
    if tf.shape(y_true_f)[-1] <= 1 or tf.shape(y_true_f)[0] <=1 : # Covariance not well-defined
        covariance_loss_tracker.assign(0.0)
        return 0.0

    # Using tfp.stats.covariance if available, or manual calculation
    # For simplicity, let's use a placeholder for the actual calculation
    # cov_true = tfp.stats.covariance(y_true_f)
    # cov_pred = tfp.stats.covariance(y_pred_f)
    # cov_loss = tf.norm(cov_true - cov_pred, ord='fro', axis=[-2,-1])
    cov_loss_value = tf.constant(0.0, dtype=tf.float32) # Replace with actual calculation
    
    covariance_loss_tracker.assign(cov_loss_value)
    return cov_loss_value * config.get('covariance_loss_weight', 0.0)


class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin):
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.autoencoder_model = None # This will be the single-step CVAE model
        self.encoder_model = None # Component model from encoder_plugin
        self.decoder_model = None # Component model from decoder_plugin
        self.model = None # Points to self.autoencoder_model
        print(f"[AutoencoderManager] Initialized for Sequential CVAE components.")

    def build_autoencoder(self, config):
        try:
            print("[build_autoencoder] Starting to build single-step CVAE with Conv1D Encoder and 6-feature Decoder...")

            # Get dimensions from config
            # For Encoder Input (Windowed Data)
            window_size = config.get('window_size')
            input_features_per_step = config.get('input_features_per_step') # Full features in the input window

            # For CVAE context and latent space
            rnn_hidden_dim = config.get('rnn_hidden_dim') # For h_context
            conditioning_dim = config.get('conditioning_dim') # For general conditions_t (e.g., previous 6 targets)
            latent_dim = config.get('latent_dim')

            # For Decoder Output (and CVAE target)
            output_feature_dim = 6 # Hardcoded as per requirement for OHLC + 2 derived

            if not all(v is not None for v in [window_size, input_features_per_step, rnn_hidden_dim, conditioning_dim, latent_dim]):
                raise ValueError(
                    "Config must provide 'window_size', 'input_features_per_step', 'rnn_hidden_dim', "
                    "'conditioning_dim', and 'latent_dim'."
                )
            if not isinstance(window_size, int) or window_size <= 0:
                raise ValueError(f"'window_size' must be a positive integer. Got: {window_size}")
            if not isinstance(input_features_per_step, int) or input_features_per_step <= 0:
                raise ValueError(f"'input_features_per_step' must be a positive integer. Got: {input_features_per_step}")


            # 1. Configure and get Encoder (Processes windowed input)
            self.encoder_plugin.configure_model_architecture(
                window_size=window_size,
                input_features_per_step=input_features_per_step,
                rnn_hidden_dim=rnn_hidden_dim,
                conditioning_dim=conditioning_dim,
                latent_dim=latent_dim,
                config=config
            )
            self.encoder_model = getattr(self.encoder_plugin, 'inference_network_model', None)
            if self.encoder_model is None:
                raise ValueError("Encoder plugin failed to build its model (inference_network_model).")
            print("[build_autoencoder] Encoder component (Conv1D-based) built.")
            self.encoder_model.summary(line_length=120)

            # 2. Configure and get Decoder (Generates 6 features)
            self.decoder_plugin.configure_model_architecture(
                latent_dim=latent_dim,
                rnn_hidden_dim=rnn_hidden_dim,
                conditioning_dim=conditioning_dim,
                output_feature_dim=output_feature_dim, # Should be 6, plugin enforces this
                config=config
            )
            self.decoder_model = getattr(self.decoder_plugin, 'generative_network_model', None)
            if self.decoder_model is None:
                raise ValueError("Decoder plugin failed to build its model (generative_network_model).")
            print(f"[build_autoencoder] Decoder component (outputting {output_feature_dim} features) built.")
            self.decoder_model.summary(line_length=120)

            # 3. Define Inputs for the combined Single-Step CVAE model
            # Input for the encoder part of CVAE
            input_x_window = Input(shape=(window_size, input_features_per_step), name="cvae_input_x_window")
            # Context inputs, shared by encoder and decoder logic
            input_h_context = Input(shape=(rnn_hidden_dim,), name="cvae_input_h_context")
            input_conditions_t = Input(shape=(conditioning_dim,), name="cvae_input_conditions_t")

            # 4. Pass inputs through Encoder
            # Encoder plugin expects [x_window, h_context, conditions_t]
            z_mean, z_log_var = self.encoder_model([input_x_window, input_h_context, input_conditions_t])

            # 5. Sampling Layer
            def sampling(args):
                z_mean_sample, z_log_var_sample = args
                batch = tf.shape(z_mean_sample)[0]
                dim = tf.shape(z_mean_sample)[1]
                epsilon = K.random_normal(shape=(batch, dim))
                return z_mean_sample + K.exp(0.5 * z_log_var_sample) * epsilon
            
            z = Lambda(sampling, output_shape=(latent_dim,), name='cvae_sampling_z')([z_mean, z_log_var])

            # 6. Pass z and context to Decoder
            # Decoder plugin expects [z, h_context, conditions_t]
            reconstruction = self.decoder_model([z, input_h_context, input_conditions_t]) # Output shape (batch, 6)

            # 7. Create the Single-Step CVAE Model
            # Define model with named outputs
            self.autoencoder_model = Model(
                inputs=[input_x_window, input_h_context, input_conditions_t], # CVAE takes the window as x_input
                outputs={
                    'reconstruction_output': reconstruction, # This will be y_pred in the reconstruction_loss_fn
                    'z_mean_output': z_mean,                 # For KL metric
                    'z_log_var_output': z_log_var            # For KL metric
                },
                name="windowed_input_cvae_6_features_out"
            )
            self.model = self.autoencoder_model 
            
            # KL divergence will be calculated within the main loss function
            # self.autoencoder_model.add_loss(weighted_kl_loss_for_model) # This line is removed

            print("[build_autoencoder] Single-step CVAE model assembled.")
            self.autoencoder_model.summary(line_length=150)

            # 8. Compile the CVAE Model
            adam_optimizer = Adam(
                learning_rate=config.get('learning_rate', 0.0001),
                beta_1=config.get('beta_1', 0.9),
                beta_2=config.get('beta_2', 0.999),
                epsilon=config.get('epsilon', 1e-7),
                amsgrad=config.get('amsgrad', False)
            )

            # Loss function wrapper
            # y_true is the target for 'reconstruction_output' (batch_size, 6)
            # y_pred is a dictionary of all model outputs:
            # {'reconstruction_output': tensor, 'z_mean_output': tensor, 'z_log_var_output': tensor}
            def combined_cvae_loss_fn(y_true_recon_target, y_pred_outputs_dict):
                
                reconstruction_pred_f32 = tf.cast(y_pred_outputs_dict['reconstruction_output'], tf.float32)
                y_true_f32 = tf.cast(y_true_recon_target, tf.float32) # Shape (batch, 6)
                
                z_mean_pred = y_pred_outputs_dict['z_mean_output']
                z_log_var_pred = y_pred_outputs_dict['z_log_var_output']

                # Reconstruction Loss (Huber) - operates on the 6 features
                h_loss = Huber(delta=config.get('huber_delta', 1.0))(y_true_f32, reconstruction_pred_f32)
                huber_loss_tracker.assign(h_loss)
                total_loss = h_loss

                # KL Divergence
                kl_div_term = -0.5 * keras.ops.sum(1 + z_log_var_pred - keras.ops.square(z_mean_pred) - keras.ops.exp(z_log_var_pred), axis=-1)
                kl_div_term_mean = keras.ops.mean(kl_div_term)
                weighted_kl_loss = config.get('kl_beta', 1.0) * kl_div_term_mean
                kl_loss_tracker.assign(weighted_kl_loss) # Track weighted KL for monitoring consistency with loss
                total_loss += weighted_kl_loss
                
                # MMD Loss (on the 6-feature reconstruction)
                mmd_weight = config.get('mmd_weight', 0.0)
                if mmd_weight > 0:
                    # y_true_f32 and reconstruction_pred_f32 are already (batch, 6)
                    # MMD compute_mmd expects 2D: (batch, features)
                    mmd_val = compute_mmd(y_true_f32, reconstruction_pred_f32, 
                                          config.get('mmd_sigma', 1.0), 
                                          config.get('mmd_sample_size', 32))
                    weighted_mmd = mmd_weight * mmd_val
                    mmd_total.assign(weighted_mmd)
                    total_loss += weighted_mmd
                else:
                    mmd_total.assign(0.0)

                # Statistical losses (skew, kurtosis, covariance) on the 6-feature reconstruction
                skew_loss_w = config.get('skew_loss_weight', 0.0)
                if skew_loss_w > 0:
                    # Flatten for moment calculation across all 6 features and batch
                    skew_true = calculate_standardized_moment(tf.reshape(y_true_f32, [-1]), 3)
                    skew_pred = calculate_standardized_moment(tf.reshape(reconstruction_pred_f32, [-1]), 3)
                    skew_loss_val = tf.abs(skew_true - skew_pred) * skew_loss_w
                    skew_loss_tracker.assign(skew_loss_val)
                    total_loss += skew_loss_val
                else:
                    skew_loss_tracker.assign(0.0)

                kurtosis_loss_w = config.get('kurtosis_loss_weight', 0.0)
                if kurtosis_loss_w > 0:
                    kurt_true = calculate_standardized_moment(tf.reshape(y_true_f32, [-1]), 4)
                    kurt_pred = calculate_standardized_moment(tf.reshape(reconstruction_pred_f32, [-1]), 4)
                    kurt_loss_val = tf.abs(kurt_true - kurt_pred) * kurtosis_loss_w
                    kurtosis_loss_tracker.assign(kurt_loss_val)
                    total_loss += kurt_loss_val
                else:
                    kurtosis_loss_tracker.assign(0.0)
                
                cov_loss_val = covariance_loss_calc(y_true_f32, reconstruction_pred_f32, config) # Operates on (batch, 6)
                total_loss += cov_loss_val

                return total_loss

            # Metrics: y_pred will be a dict of model outputs {'reconstruction_output':..., 'z_mean_output':..., 'z_log_var_output':...}
            def mmd_metric_fn(y_true, y_pred): # y_true is the target for reconstruction_output
                return mmd_total # mmd_total is updated in reconstruction_and_stats_loss_fn
            def huber_metric_fn(y_true, y_pred): # y_true is the target for reconstruction_output
                return huber_loss_tracker # huber_loss_tracker is updated in reconstruction_and_stats_loss_fn
            
            def kl_metric_fn(y_true, y_pred): # y_pred is the dict of all model outputs
                z_mean_m = y_pred['z_mean_output']
                z_log_var_m = y_pred['z_log_var_output']
                # Calculate raw KL for metric (unweighted)
                kl_raw = keras.ops.mean(-0.5 * keras.ops.sum(1 + z_log_var_m - keras.ops.square(z_mean_m) - keras.ops.exp(z_log_var_m), axis=-1))
                # kl_loss_tracker is updated in the main loss with the *weighted* KL.
                # For metric, we might want to show raw KL or weighted.
                # If kl_loss_tracker is already storing weighted, this metric can show raw.
                # Or, if kl_loss_tracker should show raw, update it here and use raw in loss too.
                # Let's assume kl_loss_tracker (from main loss) is for the weighted one.
                # This metric will compute and return raw KL.
                return kl_raw


            def skew_metric_fn(y_true, y_pred): return skew_loss_tracker
            def kurtosis_metric_fn(y_true, y_pred): return kurtosis_loss_tracker
            def covariance_metric_fn(y_true, y_pred): return covariance_loss_tracker

            # Adapters for mae_magnitude and r2_metric
            # y_true will be the dict {'reconstruction_output': tensor}
            # y_pred will be the dict {'reconstruction_output': tensor, 'z_mean_output': ..., 'z_log_var_output': ...}
            def mae_magnitude_metric_adapter(y_true_dict, y_pred_dict):
                return mae_magnitude(y_true_dict['reconstruction_output'], y_pred_dict['reconstruction_output'])

            def r2_metric_adapter(y_true_dict, y_pred_dict):
                return r2_metric(y_true_dict['reconstruction_output'], y_pred_dict['reconstruction_output'])


            metrics_list = [
                # Keras 'mae' will be applied to 'reconstruction_output' by default when loss is a dict
                # However, to be explicit or if it causes issues, we can define it like other metrics.
                # For now, let's rely on our custom mae_magnitude_metric_adapter.
                mae_magnitude_metric_adapter, 
                r2_metric_adapter,     
                huber_metric_fn, 
                kl_metric_fn, # This will show the raw KL value
                mmd_metric_fn,
                skew_metric_fn,
                kurtosis_metric_fn,
                covariance_metric_fn
            ]

            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss={'reconstruction_output': combined_cvae_loss_fn}, # Explicitly map loss to the output
                metrics=metrics_list,
                run_eagerly=config.get('run_eagerly', False)
            )
            print("[build_autoencoder] Single-step CVAE model compiled successfully.")

        except Exception as e:
            print(f"[build_autoencoder] Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train_autoencoder(self, data_inputs, data_targets, epochs=100, batch_size=128, config=None):
        if config is None: config = {}
        
        if not self.autoencoder_model:
            raise RuntimeError("[train_autoencoder] Single-step CVAE model not built. Please call build_autoencoder first.")

        if not isinstance(data_inputs, (list, tuple)) or len(data_inputs) != 3:
            raise ValueError("data_inputs must be a list/tuple of 3 arrays: [x_window_data, h_context_data, conditions_t_data]")
        
        print(f"[train_autoencoder] Starting CVAE training.")
        print(f"Input data shapes: x_window: {data_inputs[0].shape}, h_context: {data_inputs[1].shape}, conditions_t: {data_inputs[2].shape}")
        print(f"Target data shape (6 features): {data_targets.shape}")

        if data_targets.shape[-1] != 6:
            raise ValueError(f"data_targets should have 6 features, but got shape {data_targets.shape}")

        if np.isnan(data_inputs[0]).any() or np.isnan(data_inputs[1]).any() or np.isnan(data_inputs[2]).any() or np.isnan(data_targets).any():
            raise ValueError("[train_autoencoder] Training data or targets contain NaN values.")
            
        min_delta_es = config.get("min_delta", 1e-7)
        patience_es = config.get('early_patience', 10)
        start_epoch_es = config.get('start_from_epoch', 10)
        patience_rlr = config.get("reduce_lr_patience", max(1, int(patience_es / 4)))

        callbacks_list = [
            EarlyStoppingWithPatienceCounter(monitor='val_loss', patience=patience_es, restore_best_weights=True, verbose=1, start_from_epoch=start_epoch_es, min_delta=min_delta_es),
            ReduceLROnPlateauWithCounter(monitor="val_loss", factor=0.5, patience=patience_rlr, cooldown=5, min_delta=min_delta_es, verbose=1),
            LambdaCallback(on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}"))
        ]
        if config.get('use_mmd_weight_adjustment', False) and config.get('mmd_weight', 0.0) > 0:
            callbacks_list.append(MMDWeightAdjustmentCallback(config))

        # When using a single loss function for a model with multiple outputs,
        # y should be a dictionary mapping output names to targets if Keras needs to
        # align them. However, if the loss function itself handles the dict of y_preds,
        # y_true can be the direct target for the primary output (reconstruction).
        # Keras will pass y_true as is, and y_pred as a dict of model outputs.
        history = self.autoencoder_model.fit(
            x=data_inputs, 
            y={'reconstruction_output': data_targets}, # Pass targets as a dictionary
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=1,
            callbacks=callbacks_list, 
            validation_split=config.get('validation_split', 0.2)
        )
        print(f"[train_autoencoder] CVAE Training loss: {history.history['loss'][-1] if history.history['loss'] else 'N/A'}")
        print("[train_autoencoder] CVAE Training completed.")
        return history

    def encode_data(self, per_step_encoder_inputs, config=None):
        if not self.encoder_model:
            if self.autoencoder_model and self.encoder_plugin and hasattr(self.encoder_plugin, 'inference_network_model'):
                 self.encoder_model = self.encoder_plugin.inference_network_model
            else:
                raise ValueError("[encode_data] Encoder component model not available.")
        
        # Encoder plugin expects a list: [x_window_batch, h_prev_batch, conditions_t_batch]
        if not isinstance(per_step_encoder_inputs, list) or len(per_step_encoder_inputs) != 3:
            raise ValueError("[encode_data] 'per_step_encoder_inputs' must be a list of three arrays: [x_window_batch, h_context_batch, conditions_t_batch].")
        
        print(f"[encode_data] Encoding with Conv1D encoder. Input part shapes: {[d.shape for d in per_step_encoder_inputs]}")
        z_mean, z_log_var = self.encoder_plugin.encode(per_step_encoder_inputs) 
        print(f"[encode_data] Encoder produced z_mean shape: {z_mean.shape}, z_log_var shape: {z_log_var.shape}")
        return z_mean, z_log_var

    def decode_data(self, per_step_decoder_inputs, config=None):
        if not self.decoder_model:
            if self.autoencoder_model and self.decoder_plugin and hasattr(self.decoder_plugin, 'generative_network_model'):
                self.decoder_model = self.decoder_plugin.generative_network_model
            else:
                raise ValueError("[decode_data] Decoder component model not available.")

        # Decoder plugin expects a list: [z_t_batch, h_t_batch, conditions_t_batch]
        if not isinstance(per_step_decoder_inputs, list) or len(per_step_decoder_inputs) != 3:
            raise ValueError("[decode_data] 'per_step_decoder_inputs' must be a list of three arrays: [z_t_batch, h_context_batch, conditions_t_batch].")

        print(f"[decode_data] Decoding with 6-feature decoder. Input part shapes: {[d.shape for d in per_step_decoder_inputs]}")
        reconstructed_x_t = self.decoder_plugin.decode(per_step_decoder_inputs) # Output shape (batch, 6)
        print(f"[decode_data] Decoder produced reconstructed x_t shape: {reconstructed_x_t.shape}")
        return reconstructed_x_t

    def evaluate(self, data_inputs, data_targets, dataset_name, config=None):
        if config is None: config = {}
        if not self.autoencoder_model:
            print(f"[evaluate] CVAE model evaluation skipped for {dataset_name}: Model not available.")
            return None 
        
        print(f"[evaluate] Evaluating CVAE on {dataset_name}.")
        results = self.autoencoder_model.evaluate(
            x=data_inputs, 
            y={'reconstruction_output': data_targets}, # Pass targets as a dictionary
            verbose=1,
            batch_size=config.get('batch_size', 128) 
        )
        # results is a list: [total_loss, mae_on_reconstruction, huber_metric, kl_metric, mmd_metric, ...]
        # MAE is typically the first metric after loss if 'mae' was in the metrics list at compile time
        # and Keras applies it to the first output.
        # The order in results matches the order in metrics_list during compile, plus the loss.
        metrics_names = self.autoencoder_model.metrics_names
        results_dict = dict(zip(metrics_names, results))
        
        print(f"[evaluate] CVAE {dataset_name} Evaluation Results: {results_dict}")
        return results_dict # Return the dictionary of metrics

    def save_encoder(self, file_path):
        if self.encoder_model: # Save the component model
            self.encoder_model.save(file_path)
            print(f"[save_encoder] Encoder component model saved to {file_path}")
        elif self.encoder_plugin and hasattr(self.encoder_plugin, 'inference_network_model') and self.encoder_plugin.inference_network_model:
            self.encoder_plugin.inference_network_model.save(file_path) # Save from plugin if manager's copy is not set
            print(f"[save_encoder] Encoder component model (from plugin) saved to {file_path}")
        else:
            print("[save_encoder] Encoder component model not available to save.")

    def save_decoder(self, file_path):
        if self.decoder_model: # Save the component model
            self.decoder_model.save(file_path)
            print(f"[save_decoder] Decoder component model saved to {file_path}")
        elif self.decoder_plugin and hasattr(self.decoder_plugin, 'generative_network_model') and self.decoder_plugin.generative_network_model:
            self.decoder_plugin.generative_network_model.save(file_path) # Save from plugin
            print(f"[save_decoder] Decoder component model (from plugin) saved to {file_path}")
        else:
            print("[save_decoder] Decoder component model not available to save.")

    def load_encoder(self, file_path):
        # This method should load the Keras model and also inform the plugin
        loaded_keras_model = load_model(file_path, compile=False)
        self.encoder_model = loaded_keras_model # Manager holds a direct reference
        print(f"[load_encoder] Encoder component Keras model loaded from {file_path}")

        if hasattr(self.encoder_plugin, 'load') and callable(getattr(self.encoder_plugin, 'load')):
            try:
                # Plugin's load method should set its internal model and params
                self.encoder_plugin.load(file_path, compile_model=False)
                # Ensure manager's component model is also updated if plugin re-instantiates
                if hasattr(self.encoder_plugin, 'inference_network_model'):
                    self.encoder_model = self.encoder_plugin.inference_network_model
            except Exception as e:
                print(f"[load_encoder] Error calling encoder_plugin's load method: {e}. Keras model loaded into manager.")
        elif hasattr(self.encoder_plugin, 'inference_network_model'): # Fallback
            self.encoder_plugin.inference_network_model = loaded_keras_model
            print("[load_encoder] Keras model assigned to encoder_plugin.inference_network_model by manager.")


    def load_decoder(self, file_path):
        loaded_keras_model = load_model(file_path, compile=False)
        self.decoder_model = loaded_keras_model
        print(f"[load_decoder] Decoder component Keras model loaded from {file_path}")

        if hasattr(self.decoder_plugin, 'load') and callable(getattr(self.decoder_plugin, 'load')):
            try:
                self.decoder_plugin.load(file_path, compile_model=False)
                if hasattr(self.decoder_plugin, 'generative_network_model'):
                    self.decoder_model = self.decoder_plugin.generative_network_model
            except Exception as e:
                print(f"[load_decoder] Error calling decoder_plugin's load method: {e}. Keras model loaded into manager.")
        elif hasattr(self.decoder_plugin, 'generative_network_model'): # Fallback
            self.decoder_plugin.generative_network_model = loaded_keras_model
            print("[load_decoder] Keras model assigned to decoder_plugin.generative_network_model by manager.")

    def calculate_mse(self, original_data, reconstructed_data, config=None):
        print(f"[calculate_mse] Original data shape: {original_data.shape}")
        print(f"[calculate_mse] Reconstructed data shape: {reconstructed_data.shape}")
        if original_data.shape != reconstructed_data.shape:
            try:
                if np.prod(original_data.shape) == np.prod(reconstructed_data.shape):
                    reconstructed_data = reconstructed_data.reshape(original_data.shape)
                else: raise ValueError("Shape and element count mismatch.")
            except Exception as e:
                 raise ValueError(f"Shape mismatch: original {original_data.shape} vs reconstructed {reconstructed_data.shape}. Error: {e}")
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"[calculate_mse] Calculated MSE: {mse}")
        return mse

    def calculate_mae(self, original_data, reconstructed_data, config=None):
        print(f"[calculate_mae] Original data shape: {original_data.shape}")
        print(f"[calculate_mae] Reconstructed data shape: {reconstructed_data.shape}")
        if original_data.shape != reconstructed_data.shape:
            try:
                if np.prod(original_data.shape) == np.prod(reconstructed_data.shape):
                    reconstructed_data = reconstructed_data.reshape(original_data.shape)
                else: raise ValueError("Shape and element count mismatch.")
            except Exception as e:
                raise ValueError(f"Shape mismatch: original {original_data.shape} vs reconstructed {reconstructed_data.shape}. Error: {e}")
        mae = tf.reduce_mean(tf.abs(tf.cast(original_data, tf.float32) - tf.cast(reconstructed_data, tf.float32))).numpy()
        print(f"[calculate_mae] Calculated MAE: {mae}")
        return mae





