import tensorflow as tf
import keras
from tensorflow.keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Trackers (ensure these are defined globally in this module if not already)
mmd_total = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="mmd_total_tracker")
huber_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="huber_loss_tracker")
kl_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="kl_loss_tracker")
skew_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="skew_loss_tracker")
kurtosis_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="kurtosis_loss_tracker")
covariance_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="covariance_loss_tracker")

# Dummy implementations for functions if they are not defined elsewhere in your helper
# User should have their actual implementations.
def compute_mmd(x, y, sigma=1.0, sample_size=32): 
    # Replace with actual MMD calculation
    return tf.constant(0.0, dtype=tf.float32)

def calculate_standardized_moment(data, order):
    # Replace with actual moment calculation
    mean = tf.reduce_mean(data)
    std_dev = tf.math.reduce_std(data)
    if std_dev == 0:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.reduce_mean(((data - mean) / std_dev) ** order)

def covariance_loss_calc(y_true, y_pred, cfg):
    # Replace with actual covariance loss calculation
    # Example: (1 - tfp.stats.correlation(y_true, y_pred, sample_axis=0, event_axis=-1))
    return tf.constant(0.0, dtype=tf.float32)


def combined_cvae_loss_fn(y_true_dict, y_pred_outputs_dict):
    # y_true_dict is expected to be {'reconstruction_output': actual_target_tensor}
    # y_pred_outputs_dict is {'reconstruction_output': ..., 'z_mean_output': ..., 'z_log_var_output': ...}
    
    actual_reconstruction_target = tf.cast(y_true_dict['reconstruction_output'], tf.float32)
    
    recon_pred = tf.cast(y_pred_outputs_dict['reconstruction_output'], tf.float32)
    z_mean_pred = y_pred_outputs_dict['z_mean_output']
    z_log_var_pred = y_pred_outputs_dict['z_log_var_output']

    # TODO: Retrieve these weights from a config object if possible, or define them appropriately
    kl_beta = 1.0 
    mmd_weight = 0.0 
    skew_weight = 0.0
    kurtosis_weight = 0.0
    cov_weight = 0.0
    huber_delta = 1.0

    # 1) Huber Loss
    h_loss = Huber(delta=huber_delta)(actual_reconstruction_target, recon_pred)
    huber_loss_tracker.assign(h_loss)
    total_loss = h_loss

    # 2) KL Divergence
    kl_div_term = -0.5 * keras.ops.sum(1 + z_log_var_pred - keras.ops.square(z_mean_pred) - keras.ops.exp(z_log_var_pred), axis=-1)
    kl_div_term_mean = keras.ops.mean(kl_div_term)
    kl_loss_tracker.assign(kl_beta * kl_div_term_mean) # Track weighted KL as it contributes to loss
    total_loss += kl_beta * kl_div_term_mean
    
    # 3) MMD Loss
    if mmd_weight > 0:
        mmd_val = compute_mmd(actual_reconstruction_target, recon_pred)
        mmd_total.assign(mmd_val)
        total_loss += mmd_weight * mmd_val
    else:
        mmd_total.assign(0.0)
        
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
        cov_loss_val = covariance_loss_calc(actual_reconstruction_target, recon_pred, {}) # Pass relevant config if needed
        covariance_loss_tracker.assign(cov_loss_val)
        total_loss += cov_weight * cov_loss_val
    else:
        covariance_loss_tracker.assign(0.0)

    return total_loss

def get_metrics():
    # Metrics will receive y_true_dict and y_pred_outputs_dict
    def huber_metric_fn(y_true_dict, y_pred_dict):
        return huber_loss_tracker

    def kl_metric_fn(y_true_dict, y_pred_dict):
        # This tracks the weighted KL that contributed to the loss.
        # If raw KL is desired, it needs to be calculated from y_pred_dict here.
        return kl_loss_tracker

    def mmd_metric_fn(y_true_dict, y_pred_dict):
        return mmd_total

    def skew_metric_fn(y_true_dict, y_pred_dict):
        return skew_loss_tracker

    def kurtosis_metric_fn(y_true_dict, y_pred_dict):
        return kurtosis_loss_tracker
        
    def covariance_metric_fn(y_true_dict, y_pred_dict):
        return covariance_loss_tracker

    def mae_magnitude_metric(y_true_dict, y_pred_dict):
        yt_recon = tf.cast(y_true_dict['reconstruction_output'], tf.float32)
        yp_recon = tf.cast(y_pred_dict['reconstruction_output'], tf.float32)
        return tf.reduce_mean(tf.abs(yt_recon - yp_recon))

    return [
        mae_magnitude_metric,
        huber_metric_fn,
        kl_metric_fn,
        mmd_metric_fn,
        skew_metric_fn,
        kurtosis_metric_fn,
        covariance_metric_fn
    ]

# Ensure EarlyStoppingWithPatienceCounter and ReduceLROnPlateauWithCounter are defined
# or imported if they are used in autoencoder_manager.py.
# Example definitions if they are simple wrappers:
class EarlyStoppingWithPatienceCounter(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if hasattr(self, 'wait'): # Keras 3 might not always have 'wait' exposed this way
            print(f"ES patience: {self.wait}/{self.patience}")

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if hasattr(self, 'wait'):
            print(f"RLROP patience: {self.wait}/{self.patience}")