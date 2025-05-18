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

# Dummy implementations (User should have their actual implementations)
def gaussian_kernel(x, y, sigma=1.0):
    """Gaussian kernel for MMD."""
    beta = 1. / (2. * sigma**2)
    # x_col = tf.expand_dims(x, 1) # For pairwise distances if needed, but MMD sums over all pairs
    # y_lin = tf.expand_dims(y, 0)
    # dist = tf.reduce_sum((x_col - y_lin)**2, 2)
    
    # More direct way for MMD terms:
    # For K(x,x) and K(y,y), we need pairwise distances within each set
    # For K(x,y), pairwise distances between sets
    
    # This is a simplified approach for illustration.
    # A full MMD implementation would compute K(x,x), K(y,y), and K(x,y) terms.
    # For simplicity here, let's assume x and y are batches and we want a rough measure.
    # This is NOT a full unbiased MMD estimator, but will produce non-zero values.
    
    # A common way to compute kernel values for MMD:
    # XX = tf.matmul(x, x, transpose_b=True)
    # XY = tf.matmul(x, y, transpose_b=True)
    # YY = tf.matmul(y, y, transpose_b=True)

    # diag_X = tf.linalg.diag_part(XX)
    # diag_Y = tf.linalg.diag_part(YY)

    # K_XX = tf.exp(-beta * (tf.expand_dims(diag_X, 1) - 2 * XX + tf.expand_dims(diag_X, 0)))
    # K_YY = tf.exp(-beta * (tf.expand_dims(diag_Y, 1) - 2 * YY + tf.expand_dims(diag_Y, 0)))
    # K_XY = tf.exp(-beta * (tf.expand_dims(diag_X, 1) - 2 * XY + tf.expand_dims(diag_Y, 0)))
    
    # For a simpler, biased MMD^2 estimate (often used):
    # Ensure x and y are [batch_size, feature_dim]
    # This requires careful handling of dimensions.
    # Let's use a simpler pairwise distance sum for illustration of non-zero output.
    
    # Simplified kernel application for demonstration:
    # This is not a correct MMD, but will show a non-zero value if x and y differ.
    diff = x - y
    return tf.exp(-tf.reduce_sum(diff**2, axis=-1) / (2.0 * sigma**2))


def compute_mmd(x, y, sigma=1.0, sample_size=None):
    """
    Illustrative MMD calculation using a Gaussian kernel.
    This is a simplified (biased) estimator for MMD^2.
    Replace with a more robust MMD implementation if needed.
    x, y: Tensors of shape (batch_size, feature_dim)
    sigma: Kernel bandwidth
    """
    # Ensure x and y are 2D: (batch_size, num_features)
    # The loss function receives y_true_recon_tensor and y_pred_recon_tensor
    # which are likely (batch_size, output_feature_dim) e.g. (128, 6)

    if sample_size is not None:
        # Optional: Subsample for performance if batches are very large
        # Ensure sample_size is not larger than batch_size
        batch_size_x = tf.shape(x)[0]
        batch_size_y = tf.shape(y)[0]
        
        # Ensure sample_size is valid
        current_sample_size_x = tf.minimum(sample_size, batch_size_x)
        current_sample_size_y = tf.minimum(sample_size, batch_size_y)

        idx_x = tf.random.shuffle(tf.range(batch_size_x))[:current_sample_size_x]
        idx_y = tf.random.shuffle(tf.range(batch_size_y))[:current_sample_size_y]
        x_sample = tf.gather(x, idx_x)
        y_sample = tf.gather(y, idx_y)
    else:
        x_sample = x
        y_sample = y

    # Gaussian kernel MMD (biased estimator for MMD^2)
    # K(X, X)
    xx_dist = tf.reduce_sum(tf.square(tf.expand_dims(x_sample, 1) - tf.expand_dims(x_sample, 0)), axis=-1)
    k_xx = tf.exp(-xx_dist / (2.0 * sigma**2))
    
    # K(Y, Y)
    yy_dist = tf.reduce_sum(tf.square(tf.expand_dims(y_sample, 1) - tf.expand_dims(y_sample, 0)), axis=-1)
    k_yy = tf.exp(-yy_dist / (2.0 * sigma**2))
    
    # K(X, Y)
    xy_dist = tf.reduce_sum(tf.square(tf.expand_dims(x_sample, 1) - tf.expand_dims(y_sample, 0)), axis=-1)
    k_xy = tf.exp(-xy_dist / (2.0 * sigma**2))
    
    mmd_sq = tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 2 * tf.reduce_mean(k_xy)
    
    # MMD is sqrt(MMD^2), ensure non-negative before sqrt
    mmd_val = tf.sqrt(tf.maximum(0.0, mmd_sq)) 

    print(f"[compute_mmd] Sigma={sigma}, X_sample shape: {tf.shape(x_sample)}, Y_sample shape: {tf.shape(y_sample)}, Calculated MMD: {mmd_val.numpy()}")
    return mmd_val


def calculate_standardized_moment(data, order):
    mean = tf.reduce_mean(data)
    std_dev = tf.math.reduce_std(data)
    if tf.math.equal(std_dev, 0.0): # Check for zero std_dev to avoid NaN
        return tf.constant(0.0, dtype=tf.float32)
    return tf.reduce_mean(((data - mean) / std_dev) ** order)

def covariance_loss_calc(y_true, y_pred, cfg):
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