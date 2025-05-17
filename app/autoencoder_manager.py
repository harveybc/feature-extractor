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
        huber_val = logs.get('huber_metric', logs['loss'] - logs.get('kl_metric', 0.0) - logs.get('mmd_metric', 0.0) - logs.get('skew_metric', 0.0) - logs.get('kurtosis_metric', 0.0) - logs.get('covariance_metric', 0.0))
        mmd_val   = logs.get('mmd_metric', 0.0)
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

def mae_magnitude(y_true, y_pred_list): # y_pred is now a list
    """Compute MAE on the first column (magnitude) of the reconstruction."""
    y_pred = y_pred_list[0] # Actual reconstruction
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

def r2_metric(y_true, y_pred_list): # y_pred is now a list
    """Compute R² metric on the first column (magnitude) of the reconstruction."""
    y_pred = y_pred_list[0] # Actual reconstruction
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
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
        self.encoder_model = None # Component model
        self.decoder_model = None # Component model
        self.model = None # Points to self.autoencoder_model
        print(f"[AutoencoderManager] Initialized for Sequential CVAE components.")

    def build_autoencoder(self, config): # input_shape and num_channels are now less direct
        try:
            print("[build_autoencoder] Starting to build single-step CVAE components...")

            # Get dimensions from config
            x_feature_dim = config.get('x_feature_dim')
            rnn_hidden_dim = config.get('rnn_hidden_dim') # For h_context
            conditioning_dim = config.get('conditioning_dim') # For general conditions_t
            latent_dim = config.get('latent_dim') # This was 'interface_size'

            if not all(v is not None for v in [x_feature_dim, rnn_hidden_dim, conditioning_dim, latent_dim]):
                raise ValueError(
                    "Config must provide 'x_feature_dim', 'rnn_hidden_dim', "
                    "'conditioning_dim', and 'latent_dim'."
                )

            # 1. Configure and get Encoder (Per-Step Inference Network)
            self.encoder_plugin.configure_model_architecture(
                x_feature_dim=x_feature_dim,
                rnn_hidden_dim=rnn_hidden_dim, # Encoder takes h_context
                conditioning_dim=conditioning_dim, # Encoder takes conditions_t
                latent_dim=latent_dim,
                config=config
            )
            self.encoder_model = getattr(self.encoder_plugin, 'inference_network_model', None)
            if self.encoder_model is None:
                raise ValueError("Encoder plugin failed to build its model (inference_network_model).")
            print("[build_autoencoder] Per-step inference network (encoder component) built.")
            self.encoder_model.summary(line_length=120)

            # 2. Configure and get Decoder (Per-Step Generative Network)
            self.decoder_plugin.configure_model_architecture(
                latent_dim=latent_dim,
                rnn_hidden_dim=rnn_hidden_dim, # Decoder also takes h_context
                conditioning_dim=conditioning_dim, # Decoder also takes conditions_t
                output_feature_dim=x_feature_dim, # Output should match input x_t
                config=config
            )
            self.decoder_model = getattr(self.decoder_plugin, 'generative_network_model', None)
            if self.decoder_model is None:
                raise ValueError("Decoder plugin failed to build its model (generative_network_model).")
            print("[build_autoencoder] Per-step generative network (decoder component) built.")
            self.decoder_model.summary(line_length=120)

            # 3. Define Inputs for the combined Single-Step CVAE model
            input_x_t = Input(shape=(x_feature_dim,), name="cvae_input_x_t")
            input_h_context = Input(shape=(rnn_hidden_dim,), name="cvae_input_h_context")
            input_conditions_t = Input(shape=(conditioning_dim,), name="cvae_input_conditions_t")

            # 4. Pass inputs through Encoder
            # Encoder expects [x_t, h_context, conditions_t]
            z_mean, z_log_var = self.encoder_model([input_x_t, input_h_context, input_conditions_t])

            # 5. Sampling Layer
            def sampling(args):
                z_mean_sample, z_log_var_sample = args
                batch = tf.shape(z_mean_sample)[0]
                dim = tf.shape(z_mean_sample)[1]
                epsilon = K.random_normal(shape=(batch, dim))
                return z_mean_sample + K.exp(0.5 * z_log_var_sample) * epsilon
            
            z = Lambda(sampling, output_shape=(latent_dim,), name='cvae_sampling_z')([z_mean, z_log_var])

            # 6. Pass z and context to Decoder
            # Decoder expects [z, h_context, conditions_t]
            reconstruction = self.decoder_model([z, input_h_context, input_conditions_t])

            # 7. Create the Single-Step CVAE Model
            # Outputs: reconstruction (for Huber, MMD, etc.), z_mean, z_log_var (for KL)
            self.autoencoder_model = Model(
                inputs=[input_x_t, input_h_context, input_conditions_t],
                outputs=[reconstruction, z_mean, z_log_var],
                name="single_step_cvae"
            )
            self.model = self.autoencoder_model
            print("[build_autoencoder] Single-step CVAE model assembled.")
            self.autoencoder_model.summary(line_length=150)

            # 8. Compile the CVAE Model
            adam_optimizer = Adam(
                learning_rate=config.get('learning_rate', 0.0001), # Adjusted default
                beta_1=config.get('beta_1', 0.9),
                beta_2=config.get('beta_2', 0.999),
                epsilon=config.get('epsilon', 1e-7),
                amsgrad=config.get('amsgrad', False)
            )

            # Loss function wrapper
            def cvae_combined_loss(y_true, y_pred_list):
                reconstruction_pred = y_pred_list[0]
                z_mean_pred = y_pred_list[1]
                z_log_var_pred = y_pred_list[2]

                # Cast to float32
                y_true_f32 = tf.cast(y_true, tf.float32)
                reconstruction_pred_f32 = tf.cast(reconstruction_pred, tf.float32)

                # Reconstruction Loss (Huber)
                h_loss = Huber(delta=config.get('huber_delta', 1.0))(y_true_f32, reconstruction_pred_f32)
                huber_loss_tracker.assign(h_loss)
                total_loss = h_loss

                # KL Divergence
                kl_loss = -0.5 * K.sum(1 + z_log_var_pred - K.square(z_mean_pred) - K.exp(z_log_var_pred), axis=-1)
                kl_loss = K.mean(kl_loss) # Average over batch
                kl_loss_tracker.assign(kl_loss)
                total_loss += config.get('kl_beta', 1.0) * kl_loss
                
                # MMD Loss (on reconstruction)
                mmd_weight = config.get('mmd_weight', 0.0) # Default to 0 if not specified
                if mmd_weight > 0:
                    # Reshape for MMD if necessary (MMD expects 2D: (batch, features_flat))
                    y_true_flat = tf.reshape(y_true_f32, [tf.shape(y_true_f32)[0], -1])
                    reconstruction_flat = tf.reshape(reconstruction_pred_f32, [tf.shape(reconstruction_pred_f32)[0], -1])
                    
                    mmd_val = compute_mmd(y_true_flat, reconstruction_flat, config.get('mmd_sigma', 1.0), config.get('mmd_sample_size', 32))
                    weighted_mmd = mmd_weight * mmd_val
                    mmd_total.assign(weighted_mmd) # mmd_total tracks the weighted MMD
                    total_loss += weighted_mmd
                else:
                    mmd_total.assign(0.0)

                # Other statistical losses (skew, kurtosis, covariance) on reconstruction
                # These need to be adapted to use y_true_f32 and reconstruction_pred_f32
                skew_loss_w = config.get('skew_loss_weight', 0.0)
                if skew_loss_w > 0:
                    # Example: diff in skewness
                    skew_true = calculate_standardized_moment(tf.reshape(y_true_f32, [-1]), 3)
                    skew_pred = calculate_standardized_moment(tf.reshape(reconstruction_pred_f32, [-1]), 3)
                    skew_loss_val = tf.abs(skew_true - skew_pred) * skew_loss_w
                    skew_loss_tracker.assign(skew_loss_val)
                    total_loss += skew_loss_val
                else:
                    skew_loss_tracker.assign(0.0)

                kurtosis_loss_w = config.get('kurtosis_loss_weight', 0.0)
                if kurtosis_loss_w > 0:
                    # Example: diff in kurtosis
                    kurt_true = calculate_standardized_moment(tf.reshape(y_true_f32, [-1]), 4)
                    kurt_pred = calculate_standardized_moment(tf.reshape(reconstruction_pred_f32, [-1]), 4)
                    kurt_loss_val = tf.abs(kurt_true - kurt_pred) * kurtosis_loss_w
                    kurtosis_loss_tracker.assign(kurt_loss_val)
                    total_loss += kurt_loss_val
                else:
                    kurtosis_loss_tracker.assign(0.0)
                
                # Covariance loss
                cov_loss_val = covariance_loss_calc(y_true_f32, reconstruction_pred_f32, config) # Already weighted
                # covariance_loss_tracker is updated inside covariance_loss_calc
                total_loss += cov_loss_val

                return total_loss

            # Metrics
            # Keras applies metrics by comparing y_true with the first output of the model if y_pred is a list.
            # So, 'mae' will be mae(y_true, reconstruction_pred).
            def mmd_metric_fn(y_true, y_pred_list): return mmd_total # Returns the weighted MMD
            def huber_metric_fn(y_true, y_pred_list): return huber_loss_tracker
            def kl_metric_fn(y_true, y_pred_list): return kl_loss_tracker
            def skew_metric_fn(y_true, y_pred_list): return skew_loss_tracker
            def kurtosis_metric_fn(y_true, y_pred_list): return kurtosis_loss_tracker
            def covariance_metric_fn(y_true, y_pred_list): return covariance_loss_tracker

            metrics_list = [
                'mae', # Applied to y_true vs. reconstruction_pred
                huber_metric_fn, 
                kl_metric_fn,
                mmd_metric_fn,
                skew_metric_fn,
                kurtosis_metric_fn,
                covariance_metric_fn
            ]

            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss=cvae_combined_loss,
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

        # data_inputs is expected to be a list/tuple: [x_t_numpy, h_context_numpy, conditions_t_numpy]
        # data_targets is expected to be x_t_numpy (for reconstruction loss)
        if not isinstance(data_inputs, (list, tuple)) or len(data_inputs) != 3:
            raise ValueError("data_inputs must be a list/tuple of 3 arrays: [x_t_data, h_context_data, conditions_t_data]")
        
        print(f"[train_autoencoder] Starting CVAE training.")
        print(f"Input data shapes: x_t: {data_inputs[0].shape}, h_context: {data_inputs[1].shape}, conditions_t: {data_inputs[2].shape}")
        print(f"Target data shape (x_t): {data_targets.shape}")

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
        if config.get('use_mmd_weight_adjustment', False) and config.get('mmd_weight', 0.0) > 0: # Only add if MMD is active
            callbacks_list.append(MMDWeightAdjustmentCallback(config))

        history = self.autoencoder_model.fit(
            x=data_inputs, 
            y=data_targets, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=1,
            callbacks=callbacks_list, 
            validation_split=config.get('validation_split', 0.2)
        )
        print(f"[train_autoencoder] CVAE Training loss: {history.history['loss'][-1] if history.history['loss'] else 'N/A'}")
        print("[train_autoencoder] CVAE Training completed.")
        return history

    def encode_data(self, per_step_encoder_inputs, config=None): # config might not be needed if using plugin's encode
        if not self.encoder_model: # Check if the component model is available
            # Try to get it from the main model if not directly set
            if self.autoencoder_model and self.encoder_plugin and hasattr(self.encoder_plugin, 'inference_network_model'):
                 self.encoder_model = self.encoder_plugin.inference_network_model
            else:
                raise ValueError("[encode_data] Encoder component model not available.")
        
        # The encoder_plugin's 'encode' method expects a list of inputs: [x_t, h_context, conditions_t]
        if not isinstance(per_step_encoder_inputs, list) or len(per_step_encoder_inputs) != 3:
            raise ValueError("[encode_data] 'per_step_encoder_inputs' must be a list of three arrays: [x_t_batch, h_context_batch, conditions_t_batch].")
        
        print(f"[encode_data] Encoding with per-step inference network. Input part shapes: {[d.shape for d in per_step_encoder_inputs]}")
        # Use the plugin's encode method, which calls predict on its Keras model
        z_mean, z_log_var = self.encoder_plugin.encode(per_step_encoder_inputs) 
        print(f"[encode_data] Encoder produced z_mean shape: {z_mean.shape}, z_log_var shape: {z_log_var.shape}")
        return z_mean, z_log_var

    def decode_data(self, per_step_decoder_inputs, config=None): # config might not be needed
        if not self.decoder_model: # Check if the component model is available
            if self.autoencoder_model and self.decoder_plugin and hasattr(self.decoder_plugin, 'generative_network_model'):
                self.decoder_model = self.decoder_plugin.generative_network_model
            else:
                raise ValueError("[decode_data] Decoder component model not available.")

        # The decoder_plugin's 'decode' method expects a list: [z_t, h_context, conditions_t]
        if not isinstance(per_step_decoder_inputs, list) or len(per_step_decoder_inputs) != 3:
            raise ValueError("[decode_data] 'per_step_decoder_inputs' must be a list of three arrays: [z_t_batch, h_context_batch, conditions_t_batch].")

        print(f"[decode_data] Decoding with per-step generative network. Input part shapes: {[d.shape for d in per_step_decoder_inputs]}")
        # Use the plugin's decode method
        reconstructed_x_t = self.decoder_plugin.decode(per_step_decoder_inputs)
        print(f"[decode_data] Decoder produced reconstructed x_t shape: {reconstructed_x_t.shape}")
        return reconstructed_x_t

    def evaluate(self, data_inputs, data_targets, dataset_name, config=None):
        if config is None: config = {}
        if not self.autoencoder_model:
            print(f"[evaluate] CVAE model evaluation skipped for {dataset_name}: Model not available.")
            return None # Return a single None or (None, None) as per original expectation
        
        print(f"[evaluate] Evaluating CVAE on {dataset_name}.")
        results = self.autoencoder_model.evaluate(
            x=data_inputs, 
            y=data_targets, 
            verbose=1,
            batch_size=config.get('batch_size', 128) # Use batch_size from config if available
        )
        # results is a list: [total_loss, mae_on_reconstruction, huber_metric, kl_metric, mmd_metric, ...]
        loss_val = results[0]
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





