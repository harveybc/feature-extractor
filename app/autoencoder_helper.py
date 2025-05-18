import tensorflow as tf
import keras
from tensorflow.keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Trackers
mmd_total = tf.Variable(0.0, trainable=False)
huber_loss_tracker = tf.Variable(0.0, trainable=False)
kl_loss_tracker = tf.Variable(0.0, trainable=False)
skew_loss_tracker = tf.Variable(0.0, trainable=False)
kurtosis_loss_tracker = tf.Variable(0.0, trainable=False)
covariance_loss_tracker = tf.Variable(0.0, trainable=False)

def compute_mmd(x, y, sigma=1.0, sample_size=32):
    # … your MMD implementation …
    return tf.constant(0.0)

def calculate_standardized_moment(data, order):
    # … your moment implementation …
    return tf.constant(0.0)

def covariance_loss_calc(y_true, y_pred, cfg):
    # … your covariance loss …
    return tf.constant(0.0)

def combined_cvae_loss_fn(y_true, y_pred):
    # y_pred is a dict with keys: 'reconstruction_output','z_mean_output','z_log_var_output'
    recon = tf.cast(y_pred['reconstruction_output'], tf.float32)
    z_mean = y_pred['z_mean_output']
    z_log_var = y_pred['z_log_var_output']
    y_true = tf.cast(y_true, tf.float32)

    # 1) Huber
    h = Huber(delta=1.0)(y_true, recon)
    huber_loss_tracker.assign(h)
    total = h
    
    # 2) KL
    kl = -0.5 * keras.ops.sum(1 + z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var), axis=-1)
    kl = keras.ops.mean(kl)
    kl_loss_tracker.assign(kl)
    total += kl

    # 3) MMD
    m = compute_mmd(y_true, recon)
    mmd_total.assign(m)
    total += m

    # 4) Skew/Kurtosis/Cov
    s = tf.abs(calculate_standardized_moment(tf.reshape(y_true,[-1]),3)
               - calculate_standardized_moment(tf.reshape(recon,[-1]),3))
    skew_loss_tracker.assign(s)
    total += s

    k = tf.abs(calculate_standardized_moment(tf.reshape(y_true,[-1]),4)
               - calculate_standardized_moment(tf.reshape(recon,[-1]),4))
    kurtosis_loss_tracker.assign(k)
    total += k

    c = covariance_loss_calc(y_true, recon, {})
    covariance_loss_tracker.assign(c)
    total += c

    return total

def get_metrics():
    return [
        lambda y, p: huber_loss_tracker,
        lambda y, p: kl_loss_tracker,
        lambda y, p: mmd_total,
        lambda y, p: skew_loss_tracker,
        lambda y, p: kurtosis_loss_tracker,
        lambda y, p: covariance_loss_tracker,
    ]

class EarlyStoppingWithPatienceCounter(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        print(f"ES patience left: {self.wait}")

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        print(f"RLRO patience left: {self.wait}")