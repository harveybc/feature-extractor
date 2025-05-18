import tensorflow as tf
import keras
from tensorflow.keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import numpy as np

# Global TensorFlow Variables for Callback States
# These will be updated by their respective callbacks and printed by EpochEndLogger
current_lr_tracker = tf.Variable(0.0, dtype=tf.float32, name="current_lr_tracker") # Will be updated from logs['lr']
kl_beta_callback_tracker = tf.Variable(0.0, dtype=tf.float32, name="kl_beta_callback_tracker")
es_wait_tracker = tf.Variable(0, dtype=tf.int32, name="es_wait_tracker")
es_patience_config_tracker = tf.Variable(0, dtype=tf.int32, name="es_patience_config_tracker")
es_best_value_tracker = tf.Variable(np.inf, dtype=tf.float32, name="es_best_value_tracker")
rlrop_wait_tracker = tf.Variable(0, dtype=tf.int32, name="rlrop_wait_tracker")
rlrop_patience_config_tracker = tf.Variable(0, dtype=tf.int32, name="rlrop_patience_config_tracker")

# Trackers for loss components - RENAMED for clarity
mmd_total_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="mmd_total_component_tracker") # RENAMED from mmd_total
huber_loss_component_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="huber_loss_component_tracker") # RENAMED from huber_loss_tracker
skew_loss_component_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="skew_loss_component_tracker") # RENAMED from skew_loss_tracker
kurtosis_loss_component_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="kurtosis_loss_component_tracker") # RENAMED from kurtosis_loss_tracker
covariance_loss_component_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="covariance_loss_component_tracker") # RENAMED from covariance_loss_tracker


def compute_mmd(x, y, sigma=1.0, sample_size=None):
    """
    MMD calculation using a Gaussian kernel (biased estimator for MMD^2).
    x, y: Tensors of shape (batch_size, feature_dim)
    sigma: Kernel bandwidth
    """
    # This sigma check can remain as it's likely evaluated during graph construction or eagerly if possible
    if isinstance(sigma, (float, int)) and sigma <= 1e-6: 
       # tf.print("[compute_mmd] Warning: sigma is very small or zero. Setting to 1.0.")
        sigma = 1.0
    elif tf.is_tensor(sigma): # If sigma is a tensor, use tf.cond for the check
        sigma = tf.cond(
            tf.less_equal(sigma, 1e-6),
            lambda: tf.constant(1.0, dtype=sigma.dtype),
            lambda: sigma
        )


    x_sample = x
    y_sample = y
    if sample_size is not None:
        batch_size_x = tf.shape(x)[0]
        batch_size_y = tf.shape(y)[0]
        
        current_sample_size_x = tf.minimum(sample_size, batch_size_x)
        current_sample_size_y = tf.minimum(sample_size, batch_size_y)

        # Use tf.cond for conditional sampling
        x_sample = tf.cond(
            tf.greater(current_sample_size_x, 0),
            lambda: tf.gather(x, tf.random.shuffle(tf.range(batch_size_x))[:current_sample_size_x]),
            lambda: tf.zeros([0, tf.shape(x)[-1]], dtype=x.dtype)
        )

        y_sample = tf.cond(
            tf.greater(current_sample_size_y, 0),
            lambda: tf.gather(y, tf.random.shuffle(tf.range(batch_size_y))[:current_sample_size_y]),
            lambda: tf.zeros([0, tf.shape(y)[-1]], dtype=y.dtype)
        )
    
    # --- MODIFIED EMPTY CHECK ---
    # Define functions for tf.cond
    def calculate_mmd_core():
        def pairwise_sq_distances(a, b):
            a_sum_sq = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
            b_sum_sq = tf.reduce_sum(tf.square(b), axis=1, keepdims=True)
            ab_dot = tf.matmul(a, b, transpose_b=True)
            dist_sq = a_sum_sq + tf.transpose(b_sum_sq) - 2 * ab_dot
            return tf.maximum(0.0, dist_sq)

        k_xx_dist = pairwise_sq_distances(x_sample, x_sample)
        k_yy_dist = pairwise_sq_distances(y_sample, y_sample)
        k_xy_dist = pairwise_sq_distances(x_sample, y_sample)

        # Ensure sigma is not zero before division
        safe_sigma_sq = tf.maximum(sigma**2, 1e-9) # Prevent division by zero if sigma is extremely small

        k_xx = tf.exp(-k_xx_dist / (2.0 * safe_sigma_sq))
        k_yy = tf.exp(-k_yy_dist / (2.0 * safe_sigma_sq))
        k_xy = tf.exp(-k_xy_dist / (2.0 * safe_sigma_sq))
        
        mean_k_xx = tf.reduce_mean(k_xx)
        mean_k_yy = tf.reduce_mean(k_yy)
        mean_k_xy = tf.reduce_mean(k_xy)
        
        mmd_sq_val = mean_k_xx + mean_k_yy - 2 * mean_k_xy # Renamed to mmd_sq_val
        mmd_val_calc = tf.sqrt(tf.maximum(1e-9, mmd_sq_val)) # Renamed to mmd_val_calc

       # tf.print("[compute_mmd_TF_PRINT_CORE] Sigma:", sigma, "Shapes X_s, Y_s:", tf.shape(x_sample), tf.shape(y_sample),
       #          "means k_xx, k_yy, k_xy:", mean_k_xx, mean_k_yy, mean_k_xy,
       #          "mmd_sq:", mmd_sq_val, "MMD_val:", mmd_val_calc, summarize=-1)
        
        # Check for NaN within the lambda to ensure it's handled in graph mode
        return tf.cond(
            tf.math.is_nan(mmd_val_calc),
            lambda: tf.constant(0.0, dtype=tf.float32),
            lambda: mmd_val_calc
        )

    def return_zero_mmd():
       # tf.print("[compute_mmd] Warning: Samples for MMD are empty. Returning 0.")
        return tf.constant(0.0, dtype=tf.float32)

    # Use tf.cond for the empty check
    mmd_final_val = tf.cond(
        tf.logical_or(tf.equal(tf.shape(x_sample)[0], 0), tf.equal(tf.shape(y_sample)[0], 0)),
        true_fn=return_zero_mmd,
        false_fn=calculate_mmd_core
    )
    # --- END MODIFIED EMPTY CHECK ---
    
    return mmd_final_val


def calculate_standardized_moment(data, order):
    mean = tf.reduce_mean(data)
    std_dev = tf.math.reduce_std(data)
    
    # --- MODIFIED ZERO STD_DEV CHECK ---
    def calculate_moment():
        return tf.reduce_mean(((data - mean) / tf.maximum(std_dev, 1e-9)) ** order) # Add epsilon for safety

    def return_zero_moment():
        return tf.constant(0.0, dtype=tf.float32)

    # Use tf.cond to handle the std_dev == 0 case
    moment = tf.cond(
        tf.math.less_equal(std_dev, 1e-9), # Check if std_dev is very close to zero
        true_fn=return_zero_moment,
        false_fn=calculate_moment
    )
    # --- END MODIFIED CHECK ---
    return moment

def covariance_loss_calc(y_true, y_pred, cfg):
   # tf.print("[covariance_loss_calc] Placeholder called. Returning 0.")
    return tf.constant(0.0, dtype=tf.float32)


# NEW WRAPPER FUNCTION - THIS IS THE ONE TO IMPORT
def get_reconstruction_and_stats_loss_fn(outer_config):
    """
    Wrapper function that captures the configuration and returns the actual loss function
    to be used by Keras.
    """
    def reconstruction_and_stats_loss_fn_inner(y_true_recon_tensor, y_pred_recon_tensor):
        config_to_use = outer_config 

        actual_reconstruction_target = tf.cast(y_true_recon_tensor, tf.float32)
        recon_pred = tf.cast(y_pred_recon_tensor, tf.float32)

        mmd_weight = config_to_use.get('mmd_weight', 0.0) 
        mmd_sigma = config_to_use.get('mmd_sigma', 1.0)
        mmd_sample_size = config_to_use.get('mmd_sample_size', None)

        skew_weight = config_to_use.get('skew_weight', 0.0)
        kurtosis_weight = config_to_use.get('kurtosis_weight', 0.0)
        cov_weight = config_to_use.get('cov_weight', 0.0)
        huber_delta = config_to_use.get('huber_delta', 1.0)

        h_loss = Huber(delta=huber_delta)(actual_reconstruction_target, recon_pred)
        huber_loss_component_tracker.assign(h_loss) # Use RENAMED tracker
        total_loss = h_loss
        
        if mmd_weight > 0:
            mmd_val = compute_mmd(actual_reconstruction_target, recon_pred, sigma=mmd_sigma, sample_size=mmd_sample_size)
            mmd_total_tracker.assign(mmd_val) # Use RENAMED tracker
            total_loss += mmd_weight * mmd_val
        else:
            mmd_total_tracker.assign(0.0) # Use RENAMED tracker
            
        if skew_weight > 0:
            skew_true = calculate_standardized_moment(tf.reshape(actual_reconstruction_target, [-1]), 3)
            skew_pred = calculate_standardized_moment(tf.reshape(recon_pred, [-1]), 3)
            skew_loss_val = tf.abs(skew_true - skew_pred)
            skew_loss_component_tracker.assign(skew_loss_val) # Use RENAMED tracker
            total_loss += skew_weight * skew_loss_val
        else:
            skew_loss_component_tracker.assign(0.0) # Use RENAMED tracker

        if kurtosis_weight > 0:
            kurt_true = calculate_standardized_moment(tf.reshape(actual_reconstruction_target, [-1]), 4)
            kurt_pred = calculate_standardized_moment(tf.reshape(recon_pred, [-1]), 4)
            kurt_loss_val = tf.abs(kurt_true - kurt_pred)
            kurtosis_loss_component_tracker.assign(kurt_loss_val) # Use RENAMED tracker
            total_loss += kurtosis_weight * kurt_loss_val
        else:
            kurtosis_loss_component_tracker.assign(0.0) # Use RENAMED tracker
            
        if cov_weight > 0:
            cov_loss_val = covariance_loss_calc(actual_reconstruction_target, recon_pred, config_to_use) 
            covariance_loss_component_tracker.assign(cov_loss_val) # Use RENAMED tracker
            total_loss += cov_weight * cov_loss_val
        else:
            covariance_loss_component_tracker.assign(0.0) # Use RENAMED tracker

        return total_loss
    
    return reconstruction_and_stats_loss_fn_inner

# IMPORTANT: Comment out or remove the old 'reconstruction_and_stats_loss_fn'
# if 'get_reconstruction_and_stats_loss_fn' is now the primary way to get the loss function.
#
# def reconstruction_and_stats_loss_fn(y_true_recon_tensor, y_pred_recon_tensor, config=None):
#     # ... (THIS IS THE OLD IMPLEMENTATION THAT SHOULD BE REPLACED/COMMENTED IF USING THE WRAPPER ABOVE)
#     if config is None: config = {} 
#     # ... rest of old implementation ...


def get_metrics(config=None): 
    # This is the primary Keras metric we want to see in logs and evaluate results.
    # Keras will name it based on the output name + function name, e.g., "reconstruction_out_mae".
    def mae(y_true_tensor, y_pred_tensor):
        yt_recon = tf.cast(y_true_tensor, tf.float32)
        yp_recon = tf.cast(y_pred_tensor, tf.float32)
        return tf.reduce_mean(tf.abs(yt_recon - yp_recon))
    
    # Only return actual Keras-compatible metric functions here.
    # The other values (MMD, Skew, etc.) are tracked via global tf.Variables
    # and will be printed by EpochEndLogger directly from those variables.
    metrics_to_return = [mae] 
    return metrics_to_return

class EarlyStoppingWithPatienceCounter(EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store configured patience in the global tracker when callback is initialized
        if hasattr(self, 'patience'):
             es_patience_config_tracker.assign(self.patience)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs) # Original EarlyStopping logic
        # Update global trackers with current state
        if hasattr(self, 'wait'):
            es_wait_tracker.assign(self.wait)
        if self.best is not None and np.isfinite(self.best):
            es_best_value_tracker.assign(self.best)
        # Original print statement is removed; EpochEndLogger will handle display

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store configured patience in the global tracker
        if hasattr(self, 'patience'):
            rlrop_patience_config_tracker.assign(self.patience)

    def on_epoch_end(self, epoch, logs=None):
        # It's important to get LR *before* super().on_epoch_end might change it
        # However, logs['lr'] will contain the LR for the *next* epoch if it changed.
        # For current epoch's LR, it's better to read it directly or rely on logs['lr']
        # if Keras guarantees it's the one used for the ended epoch or start of next.
        # Let's assume logs['lr'] (added by ReduceLROnPlateau base class) is sufficient.
        super().on_epoch_end(epoch, logs) # Original ReduceLROnPlateau logic
        # Update global trackers
        if hasattr(self, 'wait'):
            rlrop_wait_tracker.assign(self.wait)
        # LR is typically added to logs by the base ReduceLROnPlateau callback as 'lr'
        # Original print statement is removed

class KLAnnealingCallback(Callback):
    def __init__(self, kl_beta_start, kl_beta_end, anneal_epochs, 
                 kl_layer_instance=None, layer_name="kl_loss_adder_node", verbose=0): # Set default verbose to 0
        super(KLAnnealingCallback, self).__init__()
        self.kl_beta_start = kl_beta_start
        self.kl_beta_end = kl_beta_end
        self.anneal_epochs = anneal_epochs
        self.kl_layer_instance = kl_layer_instance 
        self.layer_name = layer_name 
        self.verbose = verbose # User can override if specific prints from this cb are needed
        self.current_kl_beta = tf.Variable(kl_beta_start, trainable=False, dtype=tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.anneal_epochs:
            new_beta = self.kl_beta_start + (self.kl_beta_end - self.kl_beta_start) * (epoch / float(self.anneal_epochs))
            self.current_kl_beta.assign(new_beta)
        else:
            self.current_kl_beta.assign(self.kl_beta_end)
        
        kl_beta_callback_tracker.assign(self.current_kl_beta) # Update global tracker

        target_kl_layer = None
        if self.kl_layer_instance:
            target_kl_layer = self.kl_layer_instance
        elif self.model and self.layer_name: 
            try:
                target_kl_layer = self.model.get_layer(self.layer_name)
            except ValueError:
                if self.verbose > 0 and epoch == 0: # Infrequent tf.print for setup is okay
                   tf.print(f"\nKLAnnealingCallback: Layer '{self.layer_name}' not found by name (fallback).")
        
        if target_kl_layer:
            if hasattr(target_kl_layer, 'kl_beta') and isinstance(target_kl_layer.kl_beta, tf.Variable):
                target_kl_layer.kl_beta.assign(self.current_kl_beta)
            elif hasattr(target_kl_layer, 'kl_beta'): 
                # This case might be problematic if kl_beta is not a tf.Variable.
                # For safety, ensure KLDivergenceLayer.kl_beta is always a tf.Variable.
                target_kl_layer.kl_beta = self.current_kl_beta.numpy() 
            else:
                if self.verbose > 0 and epoch == 0:
                   tf.print(f"\nKLAnnealingCallback: Layer '{target_kl_layer.name}' does not have 'kl_beta' attribute or it's not assignable.")
        else:
            if self.verbose > 0 and epoch == 0: 
               tf.print(f"\nKLAnnealingCallback: KL divergence layer not found. KL beta will not be annealed by this callback.")

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['kl_beta_val'] = self.current_kl_beta.numpy() # Add to logs, use a distinct key
        # Original print statement is removed

# New callback for consolidated epoch-end logging
class EpochEndLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_items = [f"Epoch {epoch+1}"]

        # Standard Keras metrics from logs
        if 'loss' in logs: log_items.append(f"loss: {logs['loss']:.4f}")
        
        # MAE: Keras prepends output name. Our output is 'reconstruction_out'. Metric fn is 'mae'.
        # So, the key in logs should be 'reconstruction_out_mae'.
        mae_key = 'reconstruction_out_mae' 
        if mae_key in logs: 
            log_items.append(f"mae: {logs[mae_key]:.4f}")
        elif 'mae' in logs: # Fallback for simpler 'mae' key if Keras uses it for some reason
            log_items.append(f"mae: {logs['mae']:.4f}")
        # else: MAE not found in logs, will not be printed for this epoch

        if 'val_loss' in logs: log_items.append(f"val_loss: {logs['val_loss']:.4f}")
        
        val_mae_key = f"val_{mae_key}" # Expected "val_reconstruction_out_mae"
        if val_mae_key in logs:
            log_items.append(f"val_mae: {logs[val_mae_key]:.4f}")
        elif 'val_mae' in logs: # Fallback for simpler 'val_mae'
            log_items.append(f"val_mae: {logs['val_mae']:.4f}")
        # else: Val_MAE not found in logs, will not be printed

        # Learning rate
        current_lr_val_str = "N/A"
        if 'lr' in logs: # ReduceLROnPlateau adds 'lr' to logs
            current_lr_val_str = f"{logs['lr']:.7f}"
        elif hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'learning_rate'):
            try: 
                lr_val_obj = self.model.optimizer.learning_rate
                if isinstance(lr_val_obj, tf.Variable):
                    current_lr_val_str = f"{lr_val_obj.numpy():.7f}"
                elif callable(lr_val_obj): # If it's a LearningRateSchedule
                     # Get current step, might need a global step counter if not using iterations_per_epoch
                    current_step = self.model.optimizer.iterations.numpy() # Get current iterations
                    current_lr_val_str = f"{lr_val_obj(current_step).numpy():.7f}"
                else: # Assuming it's a direct float value
                    current_lr_val_str = f"{tf.keras.backend.get_value(lr_val_obj):.7f}"
            except Exception as e:
                # print(f"Debug: Could not retrieve LR directly: {e}") # Optional debug
                pass # Keep N/A
        log_items.append(f"lr: {current_lr_val_str}")

        # KL Beta (from global tracker, updated by KLAnnealingCallback)
        log_items.append(f"kl_beta: {kl_beta_callback_tracker.numpy():.6f}")
        
        # Log other tracked components directly from global tf.Variables (using RENAMED trackers)
        log_items.append(f"huber_c: {huber_loss_component_tracker.numpy():.4f}")
        log_items.append(f"mmd_c: {mmd_total_tracker.numpy():.4f}") # RENAMED
        log_items.append(f"skew_c: {skew_loss_component_tracker.numpy():.4f}") # RENAMED
        log_items.append(f"kurt_c: {kurtosis_loss_component_tracker.numpy():.4f}") # RENAMED
        # covariance_loss_component_tracker is also available if needed

        # Early Stopping Info
        es_patience_val = es_patience_config_tracker.numpy()
        if es_patience_val > 0: # Only print if ES is active (patience configured)
            best_val_np = es_best_value_tracker.numpy()
            best_val_str = f"{best_val_np:.4f}" if np.isfinite(best_val_np) else "N/A"
            log_items.append(f"ES_wait: {es_wait_tracker.numpy()}/{es_patience_val}")
            log_items.append(f"ES_best: {best_val_str}")

        # Reduce LR Info
        rlrop_patience_val = rlrop_patience_config_tracker.numpy()
        if rlrop_patience_val > 0: # Only print if RLROP is active (patience configured)
            log_items.append(f"RLROP_wait: {rlrop_wait_tracker.numpy()}/{rlrop_patience_val}")
        
        print(" - ".join(log_items)) # Standard Python print