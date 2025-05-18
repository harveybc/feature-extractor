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
        super().on_epoch_end(epoch, logs)
        if logs is None:
            logs = {}
        
        # self.best should hold the best value of the monitored quantity
        # It's initialized to np.inf or -np.inf by the parent class.
        # If the monitored metric (e.g., val_loss) is never available, self.best might remain np.inf or be None.
        best_val = self.best 
        
        if best_val is None or best_val == np.inf or best_val == -np.inf:
            best_loss_info = f"Best {self.monitor}: N/A"
        else:
            try:
                best_loss_info = f"Best {self.monitor}: {best_val:.6f}"
            except TypeError: # Safeguard if best_val is unexpectedly not format-able
                best_loss_info = f"Best {self.monitor}: ErrorFormatting"

        patience_info = ""
        if hasattr(self, 'wait'):
            patience_info = f"ES patience: {self.wait}/{self.patience}"
        
        print(f" - {patience_info} - {best_loss_info}", end="")

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        # MODIFIED: Get learning rate value correctly for Keras 3
        old_lr_variable = self.model.optimizer.learning_rate
        if hasattr(old_lr_variable, 'numpy'):
            old_lr = old_lr_variable.numpy()
        else: # Fallback if it's somehow a direct float (less common for new optimizers)
            old_lr = old_lr_variable 

        super().on_epoch_end(epoch, logs) # This will potentially change the LR

        new_lr_variable = self.model.optimizer.learning_rate
        if hasattr(new_lr_variable, 'numpy'):
            new_lr = new_lr_variable.numpy()
        else:
            new_lr = new_lr_variable
        
        patience_info = ""
        if hasattr(self, 'wait'):
            patience_info = f"RLROP patience: {self.wait}/{self.patience}"

        lr_info = f"LR: {new_lr:.7f}"
        if new_lr < old_lr:
            lr_info = f"LR reduced to: {new_lr:.7f}"
            
        print(f" - {patience_info} - {lr_info}", end="")


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
            logs['kl_beta'] = self.current_kl_beta.numpy() 
        if self.verbose > 0 : 
             print(f" - kl_beta: {self.current_kl_beta.numpy():.6f}", end="")