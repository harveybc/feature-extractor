import tensorflow as tf
import keras
from tensorflow.keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Trackers
mmd_total = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="mmd_total_tracker")
huber_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="huber_loss_tracker")
# kl_loss_tracker will not be updated here if KL metric is handled by the layer
# kl_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="kl_loss_tracker") 
skew_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="skew_loss_tracker")
kurtosis_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="kurtosis_loss_tracker")
covariance_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="covariance_loss_tracker")

# Dummy implementations (User should have their actual implementations)
def compute_mmd(x, y, sigma=1.0, sample_size=32): 
    return tf.constant(0.0, dtype=tf.float32)

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
        mmd_val = compute_mmd(actual_reconstruction_target, recon_pred)
        mmd_total.assign(mmd_val)
        total_loss += mmd_weight * mmd_val
    else:
        mmd_total.assign(0.0) # Ensure tracker is zero if not used
        
    # 4) Skewness Loss
    if skew_weight > 0:
        skew_true = calculate_standardized_moment(tf.reshape(actual_reconstruction_target, [-1]), 3)
        skew_pred = calculate_standardized_moment(tf.reshape(recon_pred, [-1]), 3)
        skew_loss_val = tf.abs(skew_true - skew_pred)
        skew_loss_tracker.assign(skew_loss_val)
        total_loss += skew_weight * skew_loss_val
    else:
        skew_loss_tracker.assign(0.0)

    # 5) Kurtosis Loss
    if kurtosis_weight > 0:
        kurt_true = calculate_standardized_moment(tf.reshape(actual_reconstruction_target, [-1]), 4)
        kurt_pred = calculate_standardized_moment(tf.reshape(recon_pred, [-1]), 4)
        kurt_loss_val = tf.abs(kurt_true - kurt_pred)
        kurtosis_loss_tracker.assign(kurt_loss_val)
        total_loss += kurtosis_weight * kurt_loss_val
    else:
        kurtosis_loss_tracker.assign(0.0)
        
    # 6) Covariance Loss
    if cov_weight > 0:
        # Pass relevant part of config if covariance_loss_calc needs it
        cov_loss_val = covariance_loss_calc(actual_reconstruction_target, recon_pred, config) 
        covariance_loss_tracker.assign(cov_loss_val)
        total_loss += cov_weight * cov_loss_val
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

    return [
        mae_magnitude_metric, 
        huber_metric_fn,    
        # kl_metric_fn, # Removed
        mmd_metric_fn,
        skew_metric_fn,
        kurtosis_metric_fn,
        covariance_metric_fn
    ]

# Callbacks
class EarlyStoppingWithPatienceCounter(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if hasattr(self, 'wait'): 
            print(f"ES patience: {self.wait}/{self.patience}")

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if hasattr(self, 'wait'):
            print(f"RLROP patience: {self.wait}/{self.patience}")