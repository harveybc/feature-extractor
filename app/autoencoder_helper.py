import tensorflow as tf
# import keras # Remove direct keras import if not strictly needed for something tf.keras doesn't cover
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback # Changed to tensorflow.keras
import numpy as np
import sys

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


def compute_mmd(x, y, sigma=1.0, sample_size=1000):
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.
    Now handles both 2D and 3D tensors by flattening appropriately.
    """
    # MODIFIED: Handle 3D tensors by reshaping to 2D for MMD computation
    original_x_shape = tf.shape(x)
    original_y_shape = tf.shape(y)
    
    # If 3D (batch, time, features), reshape to 2D (batch*time, features)
    if len(x.shape) == 3:
        x = tf.reshape(x, [-1, tf.shape(x)[-1]])  # (batch*time, features)
    if len(y.shape) == 3:
        y = tf.reshape(y, [-1, tf.shape(y)[-1]])  # (batch*time, features)
    
    # Ensure we have enough samples for MMD computation
    x_samples = tf.shape(x)[0]
    y_samples = tf.shape(y)[0]
    
    # Use minimum of available samples and requested sample_size
    actual_sample_size = tf.minimum(tf.minimum(x_samples, y_samples), sample_size)
    
    def calculate_mmd_core():
        # Sample from x and y
        x_indices = tf.random.uniform([actual_sample_size], 0, x_samples, dtype=tf.int32)
        y_indices = tf.random.uniform([actual_sample_size], 0, y_samples, dtype=tf.int32)
        
        x_sample = tf.gather(x, x_indices, axis=0)
        y_sample = tf.gather(y, y_indices, axis=0)
        
        # Compute kernel matrices
        k_xx_dist = pairwise_sq_distances(x_sample, x_sample)
        k_yy_dist = pairwise_sq_distances(y_sample, y_sample)
        k_xy_dist = pairwise_sq_distances(x_sample, y_sample)
        
        # RBF kernel
        k_xx = tf.exp(-k_xx_dist / (2 * sigma**2))
        k_yy = tf.exp(-k_yy_dist / (2 * sigma**2))
        k_xy = tf.exp(-k_xy_dist / (2 * sigma**2))
        
        # MMD^2 estimate
        mmd_sq = tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 2 * tf.reduce_mean(k_xy)
        return tf.maximum(mmd_sq, 0.0)  # Ensure non-negative
    
    def return_zero():
        return tf.constant(0.0, dtype=tf.float32)
    
    # Only compute MMD if we have enough samples
    mmd_final_val = tf.cond(
        actual_sample_size >= 2,
        calculate_mmd_core,
        return_zero
    )
    
    return mmd_final_val

def pairwise_sq_distances(a, b):
    """
    Compute pairwise squared distances between rows of a and b.
    Now ensures proper 2D input handling.
    """
    # MODIFIED: Ensure inputs are 2D
    if len(a.shape) != 2:
        tf.print(f"Warning: pairwise_sq_distances received non-2D tensor a with shape {tf.shape(a)}")
        a = tf.reshape(a, [-1, tf.shape(a)[-1]])
    if len(b.shape) != 2:
        tf.print(f"Warning: pairwise_sq_distances received non-2D tensor b with shape {tf.shape(b)}")
        b = tf.reshape(b, [-1, tf.shape(b)[-1]])
    
    a_sum_sq = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)  # [N, 1]
    b_sum_sq = tf.reduce_sum(tf.square(b), axis=1, keepdims=True)  # [M, 1]
    ab_dot = tf.matmul(a, tf.transpose(b))  # [N, M]
    
    # Broadcasting: [N, 1] + [1, M] - 2*[N, M] = [N, M]
    dist_sq = a_sum_sq + tf.transpose(b_sum_sq) - 2 * ab_dot
    return tf.maximum(dist_sq, 0.0)  # Ensure non-negative due to numerical issues


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
    def reconstruction_and_stats_loss_fn_inner(actual_reconstruction_target, recon_pred, config):
        """
        Inner reconstruction loss function that handles both 2D and 3D tensors.
        """
        # MODIFIED: Handle 3D tensors by computing loss across all time steps
        if len(recon_pred.shape) == 3:
            # For 3D tensors (batch, time, features), compute MAE across time and features
            mae_loss = tf.reduce_mean(tf.abs(actual_reconstruction_target - recon_pred))
        else:
            # For 2D tensors, use original computation
            mae_loss = tf.reduce_mean(tf.abs(actual_reconstruction_target - recon_pred))
        
        # MMD computation (if enabled)
        mmd_loss = 0.0
        if config.get('use_mmd_loss', False):
            mmd_sigma = config.get('mmd_sigma', 1.0)
            mmd_sample_size = config.get('mmd_sample_size', 1000)
            mmd_weight = config.get('mmd_weight', 0.01)
            
            # compute_mmd now handles 3D tensors internally
            mmd_val = compute_mmd(actual_reconstruction_target, recon_pred, sigma=mmd_sigma, sample_size=mmd_sample_size)
            mmd_loss = mmd_weight * mmd_val
        
        # Add other loss components as needed
        perplexity_loss = 0.0
        
        total_loss = mae_loss + mmd_loss + perplexity_loss
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
    # This function is currently not directly used by AutoencoderManager._compile_model
    # as it's using the string 'mae'.
    # The function name here was 'calculate_mae_for_reconstruction'.
    def calculate_mae_for_reconstruction(y_true_tensor, y_pred_tensor): 
        tf.print("[calculate_mae_for_reconstruction] Inside MAE metric function.")
        tf.print("y_true_tensor shape:", tf.shape(y_true_tensor), "y_true_tensor dtype:", y_true_tensor.dtype)
        tf.print("y_pred_tensor shape:", tf.shape(y_pred_tensor), "y_pred_tensor dtype:", y_pred_tensor.dtype)

        # Print a few sample values
        tf.print("y_true_tensor sample (first 5):", y_true_tensor[0,:5] if tf.shape(y_true_tensor)[0] > 0 else "N/A")
        tf.print("y_pred_tensor sample (first 5):", y_pred_tensor[0,:5] if tf.shape(y_pred_tensor)[0] > 0 else "N/A")

        yt_recon = tf.cast(y_true_tensor, tf.float32)
        yp_recon = tf.cast(y_pred_tensor, tf.float32)
        tf.print("Casted yt_recon dtype:", yt_recon.dtype, "Casted yp_recon dtype:", yp_recon.dtype)

        # Check for NaNs/Infs in inputs
        tf.print("Any NaNs in yt_recon:", tf.reduce_any(tf.math.is_nan(yt_recon)))
        tf.print("Any Infs in yt_recon:", tf.reduce_any(tf.math.is_inf(yt_recon)))
        tf.print("Any NaNs in yp_recon:", tf.reduce_any(tf.math.is_nan(yp_recon)))
        tf.print("Any Infs in yp_recon:", tf.reduce_any(tf.math.is_inf(yp_recon)))

        abs_diff = tf.abs(yt_recon - yp_recon)
        tf.print("abs_diff shape:", tf.shape(abs_diff))
        tf.print("abs_diff sample (first 5):", abs_diff[0,:5] if tf.shape(abs_diff)[0] > 0 else "N/A")
        
        # Check for NaNs/Infs in abs_diff
        tf.print("Any NaNs in abs_diff:", tf.reduce_any(tf.math.is_nan(abs_diff)))
        tf.print("Any Infs in abs_diff:", tf.reduce_any(tf.math.is_inf(abs_diff)))

        mae_value = tf.reduce_mean(abs_diff)
        tf.print("Calculated MAE value:", mae_value)
        
        # Check for NaNs/Infs in final MAE
        tf.print("Is MAE value NaN:", tf.math.is_nan(mae_value))
        tf.print("Is MAE value Inf:", tf.math.is_inf(mae_value))
        
        # Defensive return if MAE is NaN to avoid issues down the line, though Keras might handle it.
        # return tf.where(tf.math.is_nan(mae_value), tf.constant(0.0, dtype=tf.float32), mae_value)
        return mae_value
    
    metrics_to_return = [calculate_mae_for_reconstruction] 
    tf.print("[get_metrics] Returning MAE function:", metrics_to_return[0].__name__)
    return metrics_to_return

class EarlyStoppingWithPatienceCounter(tf.keras.callbacks.EarlyStopping): # Changed to tf.keras
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store configured patience in the global tracker when callback is initialized
        if hasattr(self, 'patience'):
             es_patience_config_tracker.assign(self.patience)
        # Initialize best value based on mode
        if self.monitor_op == np.greater:
            es_best_value_tracker.assign(-np.inf)
        else:
            es_best_value_tracker.assign(np.inf)


    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs) # Original EarlyStopping logic
        # Update global trackers with current state
        if hasattr(self, 'wait'): # wait is the counter for patience
            es_wait_tracker.assign(self.wait)
        if self.best is not None and np.isfinite(self.best): # self.best is the best monitored quantity
            es_best_value_tracker.assign(self.best)
        # else:
            # tf.print(f"DEBUG ES: epoch {epoch}, self.best: {self.best}, self.wait: {self.wait if hasattr(self, 'wait') else 'N/A'}")


class ReduceLROnPlateauWithCounter(tf.keras.callbacks.ReduceLROnPlateau): # Changed to tf.keras
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store configured patience in the global tracker
        if hasattr(self, 'patience'):
            rlrop_patience_config_tracker.assign(self.patience)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs) # Original ReduceLROnPlateau logic
        # Update global trackers
        if hasattr(self, 'wait'): # wait is the counter for patience
            rlrop_wait_tracker.assign(self.wait)
        # else:
            # tf.print(f"DEBUG RLROP: epoch {epoch}, self.wait: {self.wait if hasattr(self, 'wait') else 'N/A'}")


class KLAnnealingCallback(tf.keras.callbacks.Callback): # Changed to tf.keras
    def __init__(self, kl_beta_start, kl_beta_end, anneal_epochs, 
                 kl_layer_instance=None, layer_name="kl_loss_adder_node", verbose=0): 
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
class EpochEndLogger(tf.keras.callbacks.Callback): # Changed to tf.keras
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_items = [f"Epoch {epoch+1}/{self.params.get('epochs', 'N/A')}"] # Add total epochs

        # Standard Keras metrics from logs
        if 'loss' in logs: log_items.append(f"loss: {logs['loss']:.5f}") # Increased precision
        
        # MAE: from the dedicated output
        mae_key = 'reconstruction_out_for_mae_calc_mean_absolute_error' 
        if mae_key in logs: 
            log_items.append(f"mae: {logs[mae_key]:.5f}") # Increased precision
        
        if 'val_loss' in logs: log_items.append(f"val_loss: {logs['val_loss']:.5f}") # Increased precision
        
        val_mae_key = f"val_{mae_key}" 
        if val_mae_key in logs:
            log_items.append(f"val_mae: {logs[val_mae_key]:.5f}") # Increased precision

        # Learning rate
        current_lr_val_str = "N/A"
        if 'lr' in logs: 
            current_lr_val_str = f"{logs['lr']:.7f}"
        elif hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'learning_rate'):
            try: 
                lr_val_obj = self.model.optimizer.learning_rate
                if isinstance(lr_val_obj, tf.Variable):
                    current_lr_val_str = f"{lr_val_obj.numpy():.7f}"
                elif callable(lr_val_obj): 
                    current_step = self.model.optimizer.iterations.numpy() 
                    current_lr_val_str = f"{lr_val_obj(current_step).numpy():.7f}"
                else: 
                    current_lr_val_str = f"{tf.keras.backend.get_value(lr_val_obj):.7f}"
            except Exception:
                pass 
        log_items.append(f"lr: {current_lr_val_str}")

        # KL Beta
        log_items.append(f"kl_b: {kl_beta_callback_tracker.numpy():.6f}") # Shorter key
        
        # Log other tracked components
        log_items.append(f"huber: {huber_loss_component_tracker.numpy():.5f}") # Shorter key
        if mmd_total_tracker.numpy() != 0.0 : log_items.append(f"mmd: {mmd_total_tracker.numpy():.5f}") # Shorter key, conditional
        if skew_loss_component_tracker.numpy() != 0.0 : log_items.append(f"skew: {skew_loss_component_tracker.numpy():.5f}") # Shorter key, conditional
        if kurtosis_loss_component_tracker.numpy() != 0.0 : log_items.append(f"kurt: {kurtosis_loss_component_tracker.numpy():.5f}") # Shorter key, conditional

        # Early Stopping Info
        es_patience_val = es_patience_config_tracker.numpy()
        if es_patience_val > 0: 
            best_val_np = es_best_value_tracker.numpy()
            best_val_str = f"{best_val_np:.5f}" if np.isfinite(best_val_np) else "N/A" # Increased precision
            log_items.append(f"ES_wait: {es_wait_tracker.numpy()}/{es_patience_val}")
            log_items.append(f"ES_best: {best_val_str}")

        # Reduce LR Info
        rlrop_patience_val = rlrop_patience_config_tracker.numpy()
        if rlrop_patience_val > 0: 
            log_items.append(f"RLROP_wait: {rlrop_wait_tracker.numpy()}/{rlrop_patience_val}")
        
        tf.print(" - ".join(log_items), output_stream=sys.stdout) # Use tf.print and stdout