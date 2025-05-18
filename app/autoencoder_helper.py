import tensorflow as tf
import keras
from tensorflow.keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import numpy as np # Ensure numpy is imported

# Trackers
mmd_total = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="mmd_total_tracker")
huber_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="huber_loss_tracker")
skew_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="skew_loss_tracker")
kurtosis_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="kurtosis_loss_tracker")
covariance_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="covariance_loss_tracker")


def compute_mmd(x, y, sigma=1.0, sample_size=None):
    """
    MMD calculation using a Gaussian kernel (biased estimator for MMD^2).
    x, y: Tensors of shape (batch_size, feature_dim)
    sigma: Kernel bandwidth
    """
    if sigma <= 1e-6: # Prevent sigma from being zero or too small
        tf.print("[compute_mmd] Warning: sigma is very small or zero. Setting to 1.0.")
        sigma = 1.0

    x_sample = x
    y_sample = y
    if sample_size is not None:
        batch_size_x = tf.shape(x)[0]
        batch_size_y = tf.shape(y)[0]
        
        current_sample_size_x = tf.minimum(sample_size, batch_size_x)
        current_sample_size_y = tf.minimum(sample_size, batch_size_y)

        if current_sample_size_x > 0:
            idx_x = tf.random.shuffle(tf.range(batch_size_x))[:current_sample_size_x]
            x_sample = tf.gather(x, idx_x)
        else:
            x_sample = tf.zeros_like(x, shape=[0, tf.shape(x)[-1]]) 

        if current_sample_size_y > 0:
            idx_y = tf.random.shuffle(tf.range(batch_size_y))[:current_sample_size_y]
            y_sample = tf.gather(y, idx_y)
        else:
            y_sample = tf.zeros_like(y, shape=[0, tf.shape(y)[-1]])
    
    if tf.shape(x_sample)[0] == 0 or tf.shape(y_sample)[0] == 0:
        tf.print("[compute_mmd] Warning: Samples for MMD are empty. Returning 0.")
        return tf.constant(0.0, dtype=tf.float32)

    # Pairwise squared Euclidean distances
    def pairwise_sq_distances(a, b):
        a_sum_sq = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
        b_sum_sq = tf.reduce_sum(tf.square(b), axis=1, keepdims=True)
        ab_dot = tf.matmul(a, b, transpose_b=True)
        dist_sq = a_sum_sq + tf.transpose(b_sum_sq) - 2 * ab_dot
        return tf.maximum(0.0, dist_sq) # Ensure non-negative

    k_xx_dist = pairwise_sq_distances(x_sample, x_sample)
    k_yy_dist = pairwise_sq_distances(y_sample, y_sample)
    k_xy_dist = pairwise_sq_distances(x_sample, y_sample)

    # Gaussian kernel
    k_xx = tf.exp(-k_xx_dist / (2.0 * sigma**2))
    k_yy = tf.exp(-k_yy_dist / (2.0 * sigma**2))
    k_xy = tf.exp(-k_xy_dist / (2.0 * sigma**2))
    
    mean_k_xx = tf.reduce_mean(k_xx)
    mean_k_yy = tf.reduce_mean(k_yy)
    mean_k_xy = tf.reduce_mean(k_xy)
    
    mmd_sq = mean_k_xx + mean_k_yy - 2 * mean_k_xy
    mmd_val = tf.sqrt(tf.maximum(1e-9, mmd_sq)) # Add epsilon for stability

    tf.print("[compute_mmd_TF_PRINT] Sigma:", sigma, "Shapes X_s, Y_s:", tf.shape(x_sample), tf.shape(y_sample),
             "means k_xx, k_yy, k_xy:", mean_k_xx, mean_k_yy, mean_k_xy,
             "mmd_sq:", mmd_sq, "MMD_val:", mmd_val, summarize=-1)
    
    if tf.math.is_nan(mmd_val):
        tf.print("[compute_mmd_TF_PRINT] MMD_val is NaN! Check inputs and sigma.")
        return tf.constant(0.0, dtype=tf.float32) # Return safe value

    return mmd_val


def calculate_standardized_moment(data, order):
    mean = tf.reduce_mean(data)
    std_dev = tf.math.reduce_std(data)
    if tf.math.equal(std_dev, 0.0): # Check for zero std_dev to avoid NaN
        return tf.constant(0.0, dtype=tf.float32)
    return tf.reduce_mean(((data - mean) / std_dev) ** order)

def covariance_loss_calc(y_true, y_pred, cfg):
    # --- THIS IS STILL A PLACEHOLDER ---
    # You MUST implement this if cov_weight > 0 in your config.
    # Example: tf.abs(tfp.stats.covariance(y_true, y_pred) - desired_covariance_or_diff)
    # Or, if you want to match the covariance matrices:
    # cov_true = tfp.stats.covariance(y_true)
    # cov_pred = tfp.stats.covariance(y_pred)
    # return tf.reduce_mean(tf.square(cov_true - cov_pred))
    tf.print("[covariance_loss_calc] Placeholder called. Returning 0.")
    return tf.constant(0.0, dtype=tf.float32)


# This is the loss function for 'reconstruction_output'
# Keras will pass the specific y_true tensor and y_pred tensor for this output.
def reconstruction_and_stats_loss_fn(y_true_recon_tensor, y_pred_recon_tensor, config=None):
    if config is None: config = {} 

    actual_reconstruction_target = tf.cast(y_true_recon_tensor, tf.float32)
    recon_pred = tf.cast(y_pred_recon_tensor, tf.float32)

    # Get weights from config or use defaults
    mmd_weight = config.get('mmd_weight', 0.0) 
    mmd_sigma = config.get('mmd_sigma', 1.0) # Get sigma for MMD
    # mmd_sample_size = config.get('mmd_sample_size', None) # Get sample_size for MMD

    skew_weight = config.get('skew_weight', 0.0)
    kurtosis_weight = config.get('kurtosis_weight', 0.0)
    cov_weight = config.get('cov_weight', 0.0)
    huber_delta = config.get('huber_delta', 1.0)

    # 1) Huber Loss (Main reconstruction loss)
    h_loss = Huber(delta=huber_delta)(actual_reconstruction_target, recon_pred)
    huber_loss_tracker.assign(h_loss) # Update tracker
    total_loss = h_loss
    
    # KL Divergence is NOT calculated here. It's added by KLDivergenceLayer in the model.
    
    # 3) MMD Loss
    if mmd_weight > 0:
        # Pass sigma (and sample_size if using) to your MMD function
        mmd_val = compute_mmd(actual_reconstruction_target, recon_pred, sigma=mmd_sigma) #, sample_size=mmd_sample_size)
        if tf.rank(mmd_val) != 0: # Ensure mmd_val is a scalar
            mmd_val = tf.reduce_mean(mmd_val) # Or appropriate reduction
        mmd_total.assign(mmd_val)
        total_loss += mmd_weight * mmd_val
        print(f"[LossFn Debug] MMD calculated: {mmd_val.numpy()}, weight: {mmd_weight}") # For debugging
    else:
        mmd_total.assign(0.0) # Ensure tracker is zero if not used
        
    # 4) Skewness Loss
    if skew_weight > 0:
        skew_true = calculate_standardized_moment(tf.reshape(actual_reconstruction_target, [-1]), 3)
        skew_pred = calculate_standardized_moment(tf.reshape(recon_pred, [-1]), 3)
        skew_loss_val = tf.abs(skew_true - skew_pred)
        skew_loss_tracker.assign(skew_loss_val)
        total_loss += skew_weight * skew_loss_val
        print(f"[LossFn Debug] Skew loss calculated: {skew_loss_val.numpy()}, weight: {skew_weight}")
    else:
        skew_loss_tracker.assign(0.0)

    # 5) Kurtosis Loss
    if kurtosis_weight > 0:
        kurt_true = calculate_standardized_moment(tf.reshape(actual_reconstruction_target, [-1]), 4)
        kurt_pred = calculate_standardized_moment(tf.reshape(recon_pred, [-1]), 4)
        kurt_loss_val = tf.abs(kurt_true - kurt_pred)
        kurtosis_loss_tracker.assign(kurt_loss_val)
        total_loss += kurtosis_weight * kurt_loss_val
        print(f"[LossFn Debug] Kurtosis loss calculated: {kurt_loss_val.numpy()}, weight: {kurtosis_weight}")
    else:
        kurtosis_loss_tracker.assign(0.0)
        
    # 6) Covariance Loss
    if cov_weight > 0:
        # Pass relevant part of config if covariance_loss_calc needs it
        cov_loss_val = covariance_loss_calc(actual_reconstruction_target, recon_pred, config) 
        if tf.rank(cov_loss_val) != 0: # Ensure cov_loss_val is a scalar
            cov_loss_val = tf.reduce_mean(cov_loss_val)
        covariance_loss_tracker.assign(cov_loss_val)
        total_loss += cov_weight * cov_loss_val
        print(f"[LossFn Debug] Covariance loss calculated: {cov_loss_val.numpy()}, weight: {cov_weight}")
    else:
        covariance_loss_tracker.assign(0.0)

    return total_loss


def get_metrics(config=None):
    if config is None: config = {}
    # kl_beta = config.get('kl_beta', 1.0) # Not needed here if KL metric is by layer

    # These metrics are associated with 'reconstruction_output', so they get tensors.
    def huber_metric_fn(y_true_tensor, y_pred_tensor):
        return huber_loss_tracker 

    # kl_metric_fn is removed from here. It will be handled by KLDivergenceLayer.add_metric()

    def mmd_metric_fn(y_true_tensor, y_pred_tensor): return mmd_total
    def skew_metric_fn(y_true_tensor, y_pred_tensor): return skew_loss_tracker
    def kurtosis_metric_fn(y_true_tensor, y_pred_tensor): return kurtosis_loss_tracker
    def covariance_metric_fn(y_true_tensor, y_pred_tensor): return covariance_loss_tracker

    def mae_magnitude_metric(y_true_tensor, y_pred_tensor):
        yt_recon = tf.cast(y_true_tensor, tf.float32)
        yp_recon = tf.cast(y_pred_tensor, tf.float32)
        return tf.reduce_mean(tf.abs(yt_recon - yp_recon))

    metrics_to_return = [mae_magnitude_metric] # Ensure MAE is always included

    # Conditionally add other metrics based on config if needed, for now, always include all
    metrics_to_return.extend([
        huber_metric_fn, mmd_metric_fn, skew_metric_fn, 
        kurtosis_metric_fn, covariance_metric_fn
    ])
    return metrics_to_return


# Callbacks
class EarlyStoppingWithPatienceCounter(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs) # Important to call parent for its logic
        # If logs is None, the super call might populate it or handle it.
        # We don't strictly need to re-initialize if super() handles it.
        # if logs is None:
        #     logs = {} 
        
        best_val = self.best
        best_loss_str = "N/A"
        # Check if best_val is a valid number before formatting
        if not (best_val is None or best_val == np.inf or best_val == -np.inf):
            try:
                best_loss_str = f"{best_val:.6f}"
            except TypeError: 
                best_loss_str = "ErrorFormatting"
        
        # Do NOT add to logs dictionary for Keras to print if it's a string Keras can't average.
        # logs[f'best_{self.monitor}'] = best_loss_str 
        
        patience_str = ""
        if hasattr(self, 'wait'):
            # Do NOT add to logs dictionary
            # logs['es_patience'] = f"{self.wait}/{self.patience}"
            patience_str = f" - ES patience: {self.wait}/{self.patience}"

        # Reinstate custom print statement
        print(f"{patience_str} - Best {self.monitor}: {best_loss_str}", end="")

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        # The parent class ReduceLROnPlateau already handles LR changes and adds 'lr' to logs.
        # Get old LR before super().on_epoch_end() potentially changes it
        old_lr_variable = self.model.optimizer.learning_rate
        if hasattr(old_lr_variable, 'numpy'):
            old_lr = old_lr_variable.numpy()
        else: 
            old_lr = old_lr_variable

        super().on_epoch_end(epoch, logs) 
        # if logs is None: # Super should handle logs
        #     logs = {}

        new_lr_variable = self.model.optimizer.learning_rate
        if hasattr(new_lr_variable, 'numpy'):
            new_lr = new_lr_variable.numpy()
        else:
            new_lr = new_lr_variable
        
        patience_str = ""
        if hasattr(self, 'wait'):
            # Do NOT add to logs dictionary
            # logs['rlrop_patience'] = f"{self.wait}/{self.patience}"
            patience_str = f" - RLROP patience: {self.wait}/{self.patience}"
        
        lr_info_str = f" - LR: {new_lr:.7f}"
        if new_lr < old_lr: # Check if LR actually changed
            lr_info_str = f" - LR reduced to: {new_lr:.7f}"

        # Reinstate custom print statement
        # Keras will automatically log the 'lr' from the logs dict if ReduceLROnPlateau updated it.
        # So we only need to print our custom patience string.
        print(f"{patience_str}{lr_info_str}", end="")


class KLAnnealingCallback(Callback):
    def __init__(self, kl_beta_start, kl_beta_end, anneal_epochs, 
                 kl_layer_instance=None, layer_name="kl_loss_adder_node", verbose=1):
        super(KLAnnealingCallback, self).__init__()
        self.kl_beta_start = kl_beta_start
        self.kl_beta_end = kl_beta_end
        self.anneal_epochs = anneal_epochs
        self.kl_layer_instance = kl_layer_instance # Store the passed instance
        self.layer_name = layer_name # Keep for fallback or reference
        self.verbose = verbose 
        self.current_kl_beta = tf.Variable(kl_beta_start, trainable=False, dtype=tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.anneal_epochs:
            new_beta = self.kl_beta_start + (self.kl_beta_end - self.kl_beta_start) * (epoch / float(self.anneal_epochs))
            self.current_kl_beta.assign(new_beta)
        else:
            self.current_kl_beta.assign(self.kl_beta_end)
        
        target_kl_layer = None
        if self.kl_layer_instance:
            target_kl_layer = self.kl_layer_instance
        elif self.model and self.layer_name: # Fallback to finding by name if instance not provided
            try:
                target_kl_layer = self.model.get_layer(self.layer_name)
            except ValueError:
                if self.verbose > 0 and epoch == 0:
                    print(f"\nKLAnnealingCallback: Layer '{self.layer_name}' not found by name (fallback).")
        
        if target_kl_layer:
            if hasattr(target_kl_layer, 'kl_beta') and isinstance(target_kl_layer.kl_beta, tf.Variable):
                target_kl_layer.kl_beta.assign(self.current_kl_beta)
                if self.verbose > 0 and epoch == 0: # Print initial beta only once
                    print(f"\nKLAnnealingCallback: Initial kl_beta set to {self.current_kl_beta.numpy():.6f} for layer '{target_kl_layer.name}'")
            elif hasattr(target_kl_layer, 'kl_beta'): 
                # This case is for when kl_beta is a simple attribute, not a tf.Variable.
                # Note: This might not trigger updates if Keras/TF optimizes graph execution.
                # It's always better for kl_beta to be a tf.Variable in the layer itself.
                target_kl_layer.kl_beta = self.current_kl_beta.numpy() 
                if self.verbose > 0 and epoch == 0:
                    print(f"\nKLAnnealingCallback: Initial kl_beta set (non-Variable) to {self.current_kl_beta.numpy():.6f} for layer '{target_kl_layer.name}'")
            else:
                if self.verbose > 0 and epoch == 0:
                    print(f"\nKLAnnealingCallback: Layer '{target_kl_layer.name}' does not have 'kl_beta' attribute or it's not assignable.")
        else:
            if self.verbose > 0 and epoch == 0: # Print only once if layer not found
                print(f"\nKLAnnealingCallback: KL divergence layer not found (neither instance provided nor found by name '{self.layer_name}'). KL beta will not be annealed by this callback.")

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['kl_beta'] = self.current_kl_beta.numpy() # This is fine, kl_beta is numerical
        
        # Reinstate custom print for kl_beta, as Keras might not show all log items by default
        # depending on verbosity settings or if other prints overwrite the line.
        if self.verbose > 0 : 
             print(f" - kl_beta: {self.current_kl_beta.numpy():.6f}", end="")