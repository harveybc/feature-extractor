import tensorflow as tf
import keras
from tensorflow.keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import numpy as np

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
    # This sigma check can remain as it's likely evaluated during graph construction or eagerly if possible
    if isinstance(sigma, (float, int)) and sigma <= 1e-6: 
        tf.print("[compute_mmd] Warning: sigma is very small or zero. Setting to 1.0.")
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

        tf.print("[compute_mmd_TF_PRINT_CORE] Sigma:", sigma, "Shapes X_s, Y_s:", tf.shape(x_sample), tf.shape(y_sample),
                 "means k_xx, k_yy, k_xy:", mean_k_xx, mean_k_yy, mean_k_xy,
                 "mmd_sq:", mmd_sq_val, "MMD_val:", mmd_val_calc, summarize=-1)
        
        # Check for NaN within the lambda to ensure it's handled in graph mode
        return tf.cond(
            tf.math.is_nan(mmd_val_calc),
            lambda: tf.constant(0.0, dtype=tf.float32),
            lambda: mmd_val_calc
        )

    def return_zero_mmd():
        tf.print("[compute_mmd] Warning: Samples for MMD are empty. Returning 0.")
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
    if tf.math.equal(std_dev, 0.0): 
        return tf.constant(0.0, dtype=tf.float32)
    return tf.reduce_mean(((data - mean) / std_dev) ** order)

def covariance_loss_calc(y_true, y_pred, cfg):
    tf.print("[covariance_loss_calc] Placeholder called. Returning 0.")
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

        tf.print("[LossFn_INNER] Using weights - MMD:", mmd_weight, "Skew:", skew_weight, 
                 "Kurtosis:", kurtosis_weight, "Cov:", cov_weight, "HuberDelta:", huber_delta, summarize=-1)
        tf.print("[LossFn_INNER] MMD params - Sigma:", mmd_sigma, "Sample Size:", mmd_sample_size, summarize=-1)

        h_loss = Huber(delta=huber_delta)(actual_reconstruction_target, recon_pred)
        huber_loss_tracker.assign(h_loss)
        total_loss = h_loss
        
        if mmd_weight > 0:
            mmd_val = compute_mmd(actual_reconstruction_target, recon_pred, sigma=mmd_sigma, sample_size=mmd_sample_size)
            if tf.rank(mmd_val) != 0: 
                mmd_val = tf.reduce_mean(mmd_val)
            mmd_total.assign(mmd_val) 
            total_loss += mmd_weight * mmd_val
            tf.print("[LossFn_INNER] MMD calculated:", mmd_val, "weighted_mmd_added:", mmd_weight * mmd_val, summarize=-1)
        else:
            mmd_total.assign(0.0)
            
        if skew_weight > 0:
            skew_true = calculate_standardized_moment(tf.reshape(actual_reconstruction_target, [-1]), 3)
            skew_pred = calculate_standardized_moment(tf.reshape(recon_pred, [-1]), 3)
            skew_loss_val = tf.abs(skew_true - skew_pred)
            skew_loss_tracker.assign(skew_loss_val) 
            total_loss += skew_weight * skew_loss_val
            tf.print("[LossFn_INNER] Skew loss calculated:", skew_loss_val, "weighted_skew_added:", skew_weight * skew_loss_val, summarize=-1)
        else:
            skew_loss_tracker.assign(0.0)

        if kurtosis_weight > 0:
            kurt_true = calculate_standardized_moment(tf.reshape(actual_reconstruction_target, [-1]), 4)
            kurt_pred = calculate_standardized_moment(tf.reshape(recon_pred, [-1]), 4)
            kurt_loss_val = tf.abs(kurt_true - kurt_pred)
            kurtosis_loss_tracker.assign(kurt_loss_val) 
            total_loss += kurtosis_weight * kurt_loss_val
            tf.print("[LossFn_INNER] Kurtosis loss calculated:", kurt_loss_val, "weighted_kurtosis_added:", kurtosis_weight * kurt_loss_val, summarize=-1)
        else:
            kurtosis_loss_tracker.assign(0.0)
            
        if cov_weight > 0:
            cov_loss_val = covariance_loss_calc(actual_reconstruction_target, recon_pred, config_to_use) 
            if tf.rank(cov_loss_val) != 0: 
                cov_loss_val = tf.reduce_mean(cov_loss_val)
            covariance_loss_tracker.assign(cov_loss_val) 
            total_loss += cov_weight * cov_loss_val
            tf.print("[LossFn_INNER] Covariance loss calculated:", cov_loss_val, "weighted_cov_added:", cov_weight * cov_loss_val, summarize=-1)
        else:
            covariance_loss_tracker.assign(0.0)

        tf.print("[LossFn_INNER] Total calculated loss for batch:", total_loss, summarize=-1)
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
    def huber_metric_fn(y_true_tensor, y_pred_tensor):
        return huber_loss_tracker 
    def mmd_metric_fn(y_true_tensor, y_pred_tensor): return mmd_total
    def skew_metric_fn(y_true_tensor, y_pred_tensor): return skew_loss_tracker
    def kurtosis_metric_fn(y_true_tensor, y_pred_tensor): return kurtosis_loss_tracker
    def covariance_metric_fn(y_true_tensor, y_pred_tensor): return covariance_loss_tracker
    def mae_magnitude_metric(y_true_tensor, y_pred_tensor):
        yt_recon = tf.cast(y_true_tensor, tf.float32)
        yp_recon = tf.cast(y_pred_tensor, tf.float32)
        return tf.reduce_mean(tf.abs(yt_recon - yp_recon))
    metrics_to_return = [
        mae_magnitude_metric, 
        huber_metric_fn, 
        mmd_metric_fn, 
        skew_metric_fn, 
        kurtosis_metric_fn, 
        covariance_metric_fn
    ]
    return metrics_to_return

# ... (rest of your Callbacks: EarlyStoppingWithPatienceCounter, ReduceLROnPlateauWithCounter, KLAnnealingCallback) ...
# Ensure Callbacks are correctly defined below
class EarlyStoppingWithPatienceCounter(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        best_val = self.best
        best_loss_str = "N/A"
        if not (best_val is None or best_val == np.inf or best_val == -np.inf):
            try:
                best_loss_str = f"{best_val:.6f}"
            except TypeError: 
                best_loss_str = "ErrorFormatting"
        
        patience_str = ""
        if hasattr(self, 'wait'):
            patience_str = f" - ES patience: {self.wait}/{self.patience}"
        print(f"{patience_str} - Best {self.monitor}: {best_loss_str}", end="")

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        old_lr_variable = self.model.optimizer.learning_rate
        if hasattr(old_lr_variable, 'numpy'):
            old_lr = old_lr_variable.numpy()
        else: 
            old_lr = old_lr_variable

        super().on_epoch_end(epoch, logs) 

        new_lr_variable = self.model.optimizer.learning_rate
        if hasattr(new_lr_variable, 'numpy'):
            new_lr = new_lr_variable.numpy()
        else:
            new_lr = new_lr_variable
        
        patience_str = ""
        if hasattr(self, 'wait'):
            patience_str = f" - RLROP patience: {self.wait}/{self.patience}"
        
        lr_info_str = f" - LR: {new_lr:.7f}"
        if new_lr < old_lr: 
            lr_info_str = f" - LR reduced to: {new_lr:.7f}"
        print(f"{patience_str}{lr_info_str}", end="")


class KLAnnealingCallback(Callback):
    def __init__(self, kl_beta_start, kl_beta_end, anneal_epochs, 
                 kl_layer_instance=None, layer_name="kl_loss_adder_node", verbose=1):
        super(KLAnnealingCallback, self).__init__()
        self.kl_beta_start = kl_beta_start
        self.kl_beta_end = kl_beta_end
        self.anneal_epochs = anneal_epochs
        self.kl_layer_instance = kl_layer_instance 
        self.layer_name = layer_name 
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
        elif self.model and self.layer_name: 
            try:
                target_kl_layer = self.model.get_layer(self.layer_name)
            except ValueError:
                if self.verbose > 0 and epoch == 0:
                    tf.print(f"\nKLAnnealingCallback: Layer '{self.layer_name}' not found by name (fallback).")
        
        if target_kl_layer:
            if hasattr(target_kl_layer, 'kl_beta') and isinstance(target_kl_layer.kl_beta, tf.Variable):
                target_kl_layer.kl_beta.assign(self.current_kl_beta)
                if self.verbose > 0 and epoch == 0: 
                    tf.print(f"\nKLAnnealingCallback: Initial kl_beta set to {self.current_kl_beta.numpy():.6f} for layer '{target_kl_layer.name}'")
            elif hasattr(target_kl_layer, 'kl_beta'): 
                target_kl_layer.kl_beta = self.current_kl_beta.numpy() 
                if self.verbose > 0 and epoch == 0:
                    tf.print(f"\nKLAnnealingCallback: Initial kl_beta set (non-Variable) to {self.current_kl_beta.numpy():.6f} for layer '{target_kl_layer.name}'")
            else:
                if self.verbose > 0 and epoch == 0:
                    tf.print(f"\nKLAnnealingCallback: Layer '{target_kl_layer.name}' does not have 'kl_beta' attribute or it's not assignable.")
        else:
            if self.verbose > 0 and epoch == 0: 
                tf.print(f"\nKLAnnealingCallback: KL divergence layer not found. KL beta will not be annealed by this callback.")

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['kl_beta'] = self.current_kl_beta.numpy() 
        
        if self.verbose > 0 : 
             print(f" - kl_beta: {self.current_kl_beta.numpy():.6f}", end="")