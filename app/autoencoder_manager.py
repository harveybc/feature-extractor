import numpy as np
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import backend as K
from keras.callbacks import LambdaCallback, ReduceLROnPlateau
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.regularizers import l2
from keras.layers import BatchNormalization, LeakyReLU
from keras.layers import MaxPooling1D, UpSampling1D
#Lambda
from keras.layers import Lambda


#set_global_policy('mixed_float16')

#define tensorflow global variable mmd_total as a float
mmd_total = tf.Variable(0.0, dtype=tf.float32, trainable=False)
# Define new global tf.Variables for additional loss components
huber_loss_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False) # To track Huber loss component
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
        # Note: With new losses, 'huber_val' will implicitly include them if not using a separate huber_metric.
        # If 'huber_metric' is available in logs, it's better to use it.
        huber_val = logs.get('huber_metric', logs['loss'] - logs.get('mmd_metric', 0.0) - logs.get('skew_metric', 0.0) - logs.get('kurtosis_metric', 0.0) - logs.get('covariance_metric', 0.0))
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

def mae_magnitude(y_true, y_pred):
    """Compute MAE on the first column (magnitude)."""
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

def r2_metric(y_true, y_pred):
    """Compute R² metric on the first column (magnitude)."""
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
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:sample_size]
    x_sample = tf.gather(x, idx)
    y_sample = tf.gather(y, idx)
    def gaussian_kernel(x, y, sigma):
        x = tf.expand_dims(x, 1)
        y = tf.expand_dims(y, 0)
        dist = tf.reduce_sum(tf.square(x - y), axis=-1)
        return tf.exp(-dist / (2.0 * sigma ** 2))
    K_xx = gaussian_kernel(x_sample, x_sample, sigma)
    K_yy = gaussian_kernel(y_sample, y_sample, sigma)
    K_xy = gaussian_kernel(x_sample, y_sample, sigma)
    return tf.reduce_mean(K_xx) + tf.reduce_mean(K_yy) - 2 * tf.reduce_mean(K_xy)


class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin):
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.autoencoder_model = None
        self.encoder_model = None
        self.decoder_model = None
        self.model = None
        print(f"[AutoencoderManager] Initialized with encoder plugin and decoder plugin")

    def build_autoencoder(self, input_shape, interface_size, config, num_channels):
        try:
            print("[build_autoencoder] Starting to build autoencoder...")

            # Determine if sliding windows are used
            use_sliding_windows = config.get('use_sliding_windows', True)

            # Configure encoder size
            self.encoder_plugin.configure_size(input_shape, interface_size, num_channels, use_sliding_windows, config)

            # Get the encoder model
            self.encoder_model = self.encoder_plugin.encoder_model
            print("[build_autoencoder] Encoder model built and compiled successfully")
            self.encoder_model.summary()

            # Use encoder plugin's stored shape_before_flatten_for_decoder (must be concrete)
            encoder_shape_before_flatten = self.encoder_plugin.shape_before_flatten_for_decoder
            if encoder_shape_before_flatten is None or None in encoder_shape_before_flatten:
                raise ValueError(f"Invalid encoder_shape_before_flatten_for_decoder: {encoder_shape_before_flatten}")
            print(f"Encoder shape_before_flatten_for_decoder: {encoder_shape_before_flatten}")

            # Configure the decoder size using that pre-flatten shape
            self.decoder_plugin.configure_size(
                interface_size,
                input_shape,
                num_channels,
                encoder_shape_before_flatten,
                use_sliding_windows,
                config
            )

            # Get the decoder model
            self.decoder_model = self.decoder_plugin.model
            print("[build_autoencoder] Decoder model built and compiled successfully")
            self.decoder_model.summary()

            # VAE reparameterization: encoder_model now outputs two tensors [z_mean, z_log_var]
            z_mean, z_log_var = self.encoder_model.output
            # sampling layer (reparam trick)
            z = Lambda(
                lambda inputs: inputs[0]
                               + K.random_normal(K.shape(inputs[0]))
                                 * K.exp(0.5 * inputs[1]),
                name="vae_sampling"
            )([z_mean, z_log_var])
            # feed sampled z to decoder
            autoencoder_output = self.decoder_model(z)

            self.autoencoder_model = Model(
                inputs=self.encoder_model.input,
                outputs=autoencoder_output,
                name="autoencoder"
            )
            self.model = self.autoencoder_model

            # Define optimizer
            adam_optimizer = Adam(
                learning_rate=config['learning_rate'],  # Set the learning rate
                beta_1=0.9,  # Default value
                beta_2=0.999,  # Default value
                epsilon=1e-7,  # Default value
                amsgrad=False,  # Default value
                #clipnorm=1.0,  # Gradient clipping
                #clipvalue=0.5  # Gradient clipping
            )

            # --- Begin Updated Loss Definition using MMD ---
            # Gaussian RBF kernel function for two sets of samples.
            def gaussian_kernel_matrix(x, y, sigma):
                # x and y are assumed to be 2D: (batch_size, features)
                # preserve original dtype for output
                orig_dtype = x.dtype
                # cast to float32 under mixed precision
                x = tf.cast(x, tf.float32)
                y = tf.cast(y, tf.float32)
                x_size = tf.shape(x)[0]
                y_size = tf.shape(y)[0]
                dim = tf.shape(x)[1]
                # Expand dimensions for pairwise distance computation.
                x_expanded = tf.reshape(x, [x_size, 1, dim])
                y_expanded = tf.reshape(y, [1, y_size, dim])
                # Compute squared L2 distance between each pair.
                squared_diff = tf.reduce_sum(tf.square(x_expanded - y_expanded), axis=2)
                # compute RBF kernel in float32 then cast back
                kernel = tf.exp(-squared_diff / (2.0 * sigma**2))
                return tf.cast(kernel, orig_dtype)

            # Compute the Maximum Mean Discrepancy (MMD) between two batches.
            def mmd_loss_term(y_true, y_pred, sigma):
                # Ensure consistent dtype for MMD computation
                y_true_casted = tf.cast(y_true, tf.float32) # Use different var names to avoid conflict
                y_pred_casted = tf.cast(y_pred, tf.float32)
                # Flatten inputs to ensure they are 2D.
                y_true_flat = tf.reshape(y_true_casted, [tf.shape(y_true_casted)[0], -1])
                y_pred_flat = tf.reshape(y_pred_casted, [tf.shape(y_pred_casted)[0], -1])
                K_xx = gaussian_kernel_matrix(y_true_flat, y_true_flat, sigma)
                K_yy = gaussian_kernel_matrix(y_pred_flat, y_pred_flat, sigma)
                K_xy = gaussian_kernel_matrix(y_true_flat, y_pred_flat, sigma)
                # Use consistent dtype for statistical computations
                dtype = K_xx.dtype
                m = tf.cast(tf.shape(y_true_flat)[0], dtype)
                n = tf.cast(tf.shape(y_pred_flat)[0], dtype)
                # Compute the unbiased MMD statistic.
                mmd = tf.reduce_sum(K_xx) / (m * m) \
                      + tf.reduce_sum(K_yy) / (n * n) \
                      - tf.constant(2.0, dtype=dtype) * tf.reduce_sum(K_xy) / (m * n)
                return mmd

            # Helper function to calculate standardized moment (skewness, kurtosis)
            def calculate_standardized_moment(data_flat, order):
                # data_flat is [batch_size, total_dims], float32
                mu = tf.reduce_mean(data_flat)
                variance = tf.math.reduce_variance(data_flat)
                sigma = tf.sqrt(variance + 1e-6) # Add epsilon for stability
                
                # Use tf.cond for conditional logic in graph mode
                def true_fn(): # Executed if sigma < 1e-5
                    return tf.constant(0.0, dtype=data_flat.dtype)
                
                def false_fn(): # Executed if sigma >= 1e-5
                    standardized_data = (data_flat - mu) / sigma
                    moment_val = tf.reduce_mean(tf.pow(standardized_data, order))
                    return moment_val
                
                return tf.cond(sigma < 1e-5, true_fn, false_fn)

            # Helper function to calculate covariance matrix and its loss
            def covariance_loss_calc(y_true_flat, y_pred_flat):
                # y_true_flat, y_pred_flat are [batch_size, D], float32
                def calculate_cov_mat(tensor): # tensor is [batch_size, D]
                    num_samples_tf = tf.cast(tf.shape(tensor)[0], tensor.dtype)
                    
                    # Use tf.cond for conditional logic
                    def if_not_enough_samples():
                        return tf.zeros((tf.shape(tensor)[1], tf.shape(tensor)[1]), dtype=tensor.dtype)

                    def if_enough_samples():
                        mean_vec = tf.reduce_mean(tensor, axis=0, keepdims=True) # [1, D]
                        centered_tensor = tensor - mean_vec # [batch_size, D]
                        # Covariance = (X_c^T * X_c) / (N-1)
                        cov_matrix = tf.matmul(centered_tensor, centered_tensor, transpose_a=True) / (num_samples_tf - 1.0)
                        return cov_matrix

                    return tf.cond(num_samples_tf <= 1, if_not_enough_samples, if_enough_samples)

                cov_true = calculate_cov_mat(y_true_flat)
                cov_pred = calculate_cov_mat(y_pred_flat)
                
                # Frobenius norm of the difference
                cov_loss = tf.norm(cov_true - cov_pred, ord='fro', axis=(-2,-1))
                return cov_loss

            # Combined loss: reconstruction (Huber) loss + weighted MMD loss + new statistical losses.
            def combined_loss(y_true, y_pred):
                # Ensure inputs are float32 for calculations
                y_true_f32 = tf.cast(y_true, tf.float32)
                y_pred_f32 = tf.cast(y_pred, tf.float32)

                # 1. Reconstruction Loss (Huber)
                huber_loss_tracker.assign(huber_loss_val) # Track for metrics

                # Prepare flattened versions for MMD, moments, and covariance
                y_true_flat_f32 = tf.reshape(y_true_f32, [tf.shape(y_true_f32)[0], -1])
                y_pred_flat_f32 = tf.reshape(y_pred_f32, [tf.shape(y_pred_f32)[0], -1])

                # 2. MMD Loss
                sigma_mmd = config.get('mmd_sigma', 1.0)
                mmd_weight = config.get('mmd_weight', 1.0)
                mmd_val = mmd_loss_term(y_true_flat_f32, y_pred_flat_f32, sigma_mmd) # mmd_loss_term uses its own casting
                weighted_mmd_loss = mmd_weight * mmd_val
                mmd_total.assign(weighted_mmd_loss) # mmd_total tracks the weighted MMD

                total_loss = huber_loss_val + weighted_mmd_loss
                
                # 3. Skewness Loss
                skew_loss_weight = config.get('skew_loss_weight', 0.0) # Default to 0 if not set
                
                def calculate_skew_loss():
                    skew_true_val = calculate_standardized_moment(y_true_flat_f32, 3)
                    skew_pred_val = calculate_standardized_moment(y_pred_flat_f32, 3)
                    skew_loss_unweighted = tf.abs(skew_true_val - skew_pred_val)
                    weighted_skew_loss = skew_loss_weight * skew_loss_unweighted
                    skew_loss_tracker.assign(weighted_skew_loss)
                    return weighted_skew_loss
                
                def no_skew_loss():
                    skew_loss_tracker.assign(0.0)
                    return tf.constant(0.0, dtype=total_loss.dtype)

                total_loss += tf.cond(tf.cast(skew_loss_weight, tf.float32) > 0.0, calculate_skew_loss, no_skew_loss)


                # 4. Kurtosis Loss
                kurtosis_loss_weight = config.get('kurtosis_loss_weight', 0.0) # Default to 0
                
                def calculate_kurtosis_loss():
                    kurt_true_val = calculate_standardized_moment(y_true_flat_f32, 4)
                    kurt_pred_val = calculate_standardized_moment(y_pred_flat_f32, 4)
                    kurtosis_loss_unweighted = tf.abs(kurt_true_val - kurt_pred_val)
                    weighted_kurtosis_loss = kurtosis_loss_weight * kurtosis_loss_unweighted
                    kurtosis_loss_tracker.assign(weighted_kurtosis_loss)
                    return weighted_kurtosis_loss

                def no_kurtosis_loss():
                    kurtosis_loss_tracker.assign(0.0)
                    return tf.constant(0.0, dtype=total_loss.dtype)
                
                total_loss += tf.cond(tf.cast(kurtosis_loss_weight, tf.float32) > 0.0, calculate_kurtosis_loss, no_kurtosis_loss)


                # 5. Covariance Matrix Loss
                covariance_loss_weight = config.get('covariance_loss_weight', 0.0) # Default to 0
                max_dim_for_cov_loss = tf.cast(config.get('max_dim_for_cov_loss', 2048), tf.int32)
                current_feature_dim = tf.shape(y_true_flat_f32)[1]
                num_samples_for_cov = tf.shape(y_true_flat_f32)[0]

                def calculate_covariance_loss():
                    cov_loss_unweighted = covariance_loss_calc(y_true_flat_f32, y_pred_flat_f32)
                    weighted_covariance_loss = tf.cast(covariance_loss_weight, y_true_flat_f32.dtype) * cov_loss_unweighted
                    covariance_loss_tracker.assign(weighted_covariance_loss)
                    return weighted_covariance_loss

                def no_covariance_loss():
                    covariance_loss_tracker.assign(0.0)
                    return tf.constant(0.0, dtype=total_loss.dtype)
                
                def print_skip_message_and_return_zero():
                   # tf.print(f"Note: Covariance loss skipped due to large feature dimension ({current_feature_dim} > {max_dim_for_cov_loss}). To enable, increase 'max_dim_for_cov_loss' in config.")
                    covariance_loss_tracker.assign(0.0)
                    return tf.constant(0.0, dtype=total_loss.dtype)

                # Condition for applying covariance loss
                condition_apply_cov_loss = tf.logical_and(
                    tf.cast(covariance_loss_weight, tf.float32) > 0.0,
                    tf.logical_and(
                        current_feature_dim <= max_dim_for_cov_loss,
                        tf.logical_and(num_samples_for_cov > 1, current_feature_dim > 0)
                    )
                )
                
                # Condition for printing skip message (if weight > 0 but other conditions fail)
                condition_print_skip = tf.logical_and(
                    tf.cast(covariance_loss_weight, tf.float32) > 0.0,
                    tf.logical_not(
                        tf.logical_and(
                            current_feature_dim <= max_dim_for_cov_loss,
                            tf.logical_and(num_samples_for_cov > 1, current_feature_dim > 0)
                        )
                    )
                )
                
                cov_loss_to_add = tf.cond(
                    condition_apply_cov_loss,
                    calculate_covariance_loss,
                    lambda: tf.cond( # Nested cond to handle the print message or just return zero
                        condition_print_skip,
                        print_skip_message_and_return_zero,
                        no_covariance_loss # This handles case where weight is 0
                    )
                )
                total_loss += cov_loss_to_add
                
                return total_loss

            # Optional: Define metrics to monitor individual loss components during training.
            def mmd_metric(y_true, y_pred):
                return mmd_total # Returns the weighted MMD value tracked globally
            
            def huber_metric(y_true, y_pred):
                return huber_loss_tracker

            def skew_metric(y_true, y_pred):
                return skew_loss_tracker

            def kurtosis_metric(y_true, y_pred):
                return kurtosis_loss_tracker

            def covariance_metric(y_true, y_pred):
                return covariance_loss_tracker
            # --- End Updated Loss Definition ---

            # Compile autoencoder with the combined loss and additional metrics.
            metrics_list = ['mae', mmd_metric, huber_metric, skew_metric, kurtosis_metric, covariance_metric]
            
            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss=combined_loss,
                metrics=metrics_list,
                run_eagerly=False # Useful for debugging complex losses, False for performance
            )
            print("[build_autoencoder] Autoencoder model built and compiled successfully")
            self.autoencoder_model.summary()
        except Exception as e:
            print(f"[build_autoencoder] Exception occurred: {e}")
            raise







    def train_autoencoder(self, data, epochs=100, batch_size=128, config=None):
        try:
            print(f"[train_autoencoder] Received data with shape: {data.shape}")

            # Determine if sliding windows are used
            use_sliding_windows = config.get('use_sliding_windows', True)

            # Check and reshape data for compatibility with Conv1D layers
            if not use_sliding_windows and len(data.shape) == 2:  # Row-by-row data (2D)
                print("[train_autoencoder] Reshaping data to add channel dimension for Conv1D compatibility...")
                data = np.expand_dims(data, axis=-1)  # Add channel dimension (num_samples, num_features, 1)
                print(f"[train_autoencoder] Reshaped data shape: {data.shape}")

            num_channels = data.shape[-1]
            input_shape = data.shape[1]
            interface_size = config.get('interface_size', 48)

            # Build autoencoder with the correct num_channels
            if not self.autoencoder_model:
                self.build_autoencoder(input_shape, interface_size, config, num_channels)

            # Validate data for NaN values before training
            if np.isnan(data).any():
                raise ValueError("[train_autoencoder] Training data contains NaN values. Please check your data preprocessing pipeline.")

            # Calculate entropy and useful information using Shannon-Hartley theorem
            #self.calculate_dataset_information(data, config)

            print(f"[train_autoencoder] Training autoencoder with data shape: {data.shape}")

            # --- Setup Callbacks ---
            min_delta_early_stopping = config.get("min_delta", config.get("min_delta", 1e-7))
            patience_early_stopping = config.get('early_patience', 10)
            start_from_epoch_es = config.get('start_from_epoch', 10)
            patience_reduce_lr = config.get("reduce_lr_patience", max(1, int(patience_early_stopping / 4)))

            # Instantiate callbacks WITHOUT ClearMemoryCallback
            # Assumes relevant Callback classes are imported/defined
            callbacks = [
                EarlyStoppingWithPatienceCounter(
                    monitor='val_loss', patience=patience_early_stopping, restore_best_weights=True,
                    verbose=1, start_from_epoch=start_from_epoch_es, min_delta=min_delta_early_stopping
                ),
                ReduceLROnPlateauWithCounter(
                    monitor="val_loss", factor=0.5, patience=patience_reduce_lr, cooldown=5, min_delta=min_delta_early_stopping, verbose=1
                ),
                LambdaCallback(on_epoch_end=lambda epoch, logs:
                            print(f"Epoch {epoch+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}"))
                #MMDWeightAdjustmentCallback(config) # You might want to re-enable and test this.
                # Removed: ClearMemoryCallback(), # <<< REMOVED THIS LINE
            ]
            if config.get('use_mmd_weight_adjustment', False): # Add a config to control this callback
                callbacks.append(MMDWeightAdjustmentCallback(config))


            # Start training with early stopping
            history = self.autoencoder_model.fit(
                data,
                data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=callbacks,
                validation_split = 0.2
            )

            # Log training loss
            print(f"[train_autoencoder] Training loss values: {history.history['loss']}")
            print("[train_autoencoder] Training completed.")
        except Exception as e:
            print(f"[train_autoencoder] Exception occurred during training: {e}")
            raise



    def calculate_dataset_information(self, data, config):
        try:
            print("[calculate_dataset_information] Calculating dataset entropy and useful information...")

            # Handle 2D and 3D data shapes
            normalized_columns = []
            if len(data.shape) == 3:  # Sliding window data
                for col in range(data.shape[-1]):
                    column_data = data[:, :, col].flatten()  # Flatten the column data
                    min_val, max_val = column_data.min(), column_data.max()
                    normalized_column = (column_data - min_val) / (max_val - min_val + 1e-7)  # Min-max normalization, add epsilon
                    normalized_columns.append(normalized_column)
            elif len(data.shape) == 2:  # Row-by-row data
                for col in range(data.shape[1]):
                    column_data = data[:, col]  # No need to flatten, already 1D
                    min_val, max_val = column_data.min(), column_data.max()
                    normalized_column = (column_data - min_val) / (max_val - min_val + 1e-7)  # Min-max normalization, add epsilon
                    normalized_columns.append(normalized_column)
            else:
                raise ValueError("[calculate_dataset_information] Unsupported data shape for processing.")

            concatenated_data = np.concatenate(normalized_columns, axis=0)
            num_samples = concatenated_data.shape[0]  # Correct number of samples is the length of concatenated vector

            # Convert concatenated data to TensorFlow tensor for acceleration
            concatenated_data_tf = tf.convert_to_tensor(concatenated_data, dtype=tf.float32)

            # Calculate signal-to-noise ratio (SNR) using TensorFlow
            mean_val = tf.reduce_mean(concatenated_data_tf)
            std_val = tf.math.reduce_std(concatenated_data_tf)
            snr = tf.cond(
                tf.greater(std_val, 1e-7), # Check against epsilon
                lambda: (mean_val / std_val) ** 2,
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

            # Retrieve dataset periodicity and calculate sampling frequency
            periodicity = config['dataset_periodicity']
            periodicity_seconds_map = {
                "1min": 60,
                "5min": 5 * 60,
                "15min": 15 * 60,
                "1h": 60 * 60,
                "4h": 4 * 60 * 60,
                "daily": 24 * 60 * 60
            }
            sampling_period_seconds = periodicity_seconds_map.get(periodicity, None)

            if sampling_period_seconds:
                sampling_frequency = tf.constant(1 / sampling_period_seconds, dtype=tf.float32)
            else:
                sampling_frequency = tf.constant(0.0, dtype=tf.float32) # Or handle as error/default

            # Calculate Shannon-Hartley channel capacity and total useful information
            channel_capacity = tf.cond(
                tf.math.logical_and(tf.greater(snr, 0), tf.greater(sampling_frequency, 0)),
                lambda: sampling_frequency * tf.math.log(1 + snr) / tf.math.log(2.0), # log base 2
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            total_information_bits = channel_capacity * num_samples * sampling_period_seconds # This seems off, num_samples is total points, not duration

            # Calculate entropy using TensorFlow histogram binning
            bins = 1000  # Increased bin count for better precision
            histogram = tf.histogram_fixed_width(concatenated_data_tf, [0.0, 1.0], nbins=bins) # Assumes data is normalized [0,1]
            histogram = tf.cast(histogram, tf.float32)
            probabilities = histogram / (tf.reduce_sum(histogram) + 1e-7) # Add epsilon
            entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-10) / tf.math.log(2.0))  # Avoid log(0), log base 2

            # Log calculated information
            print(f"[calculate_dataset_information] Calculated SNR: {snr.numpy()}")
            print(f"[calculate_dataset_information] Sampling frequency: {sampling_frequency.numpy()} Hz")
            print(f"[calculate_dataset_information] Channel capacity: {channel_capacity.numpy()} bits/second")
            #print(f"[calculate_dataset_information] Total useful information: {total_information_bits.numpy()} bits") # Re-evaluate this metric's formula if needed
            print(f"[calculate_dataset_information] Entropy: {entropy.numpy()} bits")
        except Exception as e:
            print(f"[calculate_dataset_information] Exception occurred: {e}")
            raise



    def evaluate(self, data, dataset_name, config):
        """
        Evaluate the autoencoder model on the provided dataset and calculate the MSE and MAE.

        Args:
            data (np.ndarray): Input data to evaluate (original input data).
            dataset_name (str): Name of the dataset (e.g., "Training" or "Validation").
            config (dict): Configuration dictionary.

        Returns:
            tuple: Calculated MSE and MAE for the dataset.
        """
        print(f"[evaluate] Evaluating {dataset_name} data with shape: {data.shape}")

        # Reshape data for Conv1D compatibility if sliding windows are not used
        if not config.get('use_sliding_windows', True) and len(data.shape) == 2:
            data_eval = np.expand_dims(data, axis=-1)
            print(f"[evaluate] Reshaped {dataset_name} data for Conv1D compatibility: {data_eval.shape}")
        else:
            data_eval = data

        # Evaluate the autoencoder
        # The order of results depends on model.compile(metrics=...)
        # loss is always first. Then metrics in the order provided.
        # metrics_list = ['mae', mmd_metric, huber_metric, skew_metric, kurtosis_metric, covariance_metric]
        results = self.autoencoder_model.evaluate(data_eval, data_eval, verbose=1)
        
        loss_val = results[0]
        mae_val = results[1] # Assuming 'mae' is the first in metrics_list
        # You can extract other metrics based on their order in metrics_list if needed
        # mmd_metric_val = results[2] 
        # huber_metric_val = results[3]
        # ... and so on.

        print(f"[evaluate] {dataset_name} Evaluation results - Loss: {loss_val}, MAE: {mae_val}")
        # For consistency with previous return, if MSE was expected as loss:
        # If Huber is the main reconstruction loss, loss_val is effectively a weighted sum.
        # The 'huber_metric' would give the unweighted Huber component.
        return loss_val, mae_val # Returning total loss and MAE.






    def encode_data(self, data, config):
        print(f"[encode_data] Encoding data with shape: {data.shape}")

        # Determine if sliding windows are used
        use_sliding_windows = config.get('use_sliding_windows', True)
        data_to_encode = data # Use a new variable to avoid modifying input `data` if it's passed around

        # Reshape data for sliding windows or row-by-row processing
        if use_sliding_windows:
            if 'window_size' not in config:
                 raise ValueError("[encode_data] 'window_size' must be in config if use_sliding_windows is True.")
            window_size = config['window_size']
            if window_size > data_to_encode.shape[1]: # Assuming data_to_encode is (samples, features_flat)
                raise ValueError(f"[encode_data] window_size ({window_size}) cannot be greater than the number of features ({data_to_encode.shape[1]}) in the data.")
            
            # This part assumes data is already (samples, total_features) and needs to be reshaped into (samples, window_size, num_channels)
            # This logic might need adjustment based on how data is fed initially.
            # If data is (samples, features) and features = window_size * num_channels
            if data_to_encode.shape[1] % window_size != 0:
                raise ValueError(f"[encode_data] data.shape[1] ({data_to_encode.shape[1]}) must be divisible by window_size ({window_size}) for sliding windows.")
            num_channels_calculated = data_to_encode.shape[1] // window_size
            if num_channels_calculated <= 0:
                raise ValueError("[encode_data] Invalid num_channels calculated for sliding window data.")
            
            data_to_encode = data_to_encode.reshape((data_to_encode.shape[0], window_size, num_channels_calculated))
        else:
            # For row-by-row data, add a channel dimension if it's 2D
            if len(data_to_encode.shape) == 2:
                data_to_encode = np.expand_dims(data_to_encode, axis=-1)
            # num_channels = 1 (implicitly, or data_to_encode.shape[-1])

        print(f"[encode_data] Reshaped data shape for encoding: {data_to_encode.shape}")

        # Perform encoding
        try:
            # Encoder model expects input that matches its Input layer shape
            encoded_data_parts = self.encoder_model.predict(data_to_encode)
            # For VAE, encoder_model outputs [z_mean, z_log_var]
            print(f"[encode_data] Encoded data parts (z_mean, z_log_var). z_mean shape: {encoded_data_parts[0].shape}, z_log_var shape: {encoded_data_parts[1].shape}")
            return encoded_data_parts # Return both parts
        except Exception as e:
            print(f"[encode_data] Exception occurred during encoding: {e}")
            raise ValueError("[encode_data] Failed to encode data. Please check model compatibility and data shape.")




   
    def decode_data(self, encoded_data, config):
        # For VAE, encoded_data is typically z_mean or sampled z.
        # The decoder_model expects the sampled z.
        print(f"[decode_data] Decoding data with shape: {encoded_data.shape}") # encoded_data here is z
        decoded_data = self.decoder_model.predict(encoded_data)

        # Reshaping output if not using sliding windows (to match original flat feature vector)
        # This part depends on how the original data was structured and how the decoder output is shaped.
        # The decoder is configured to output (batch, window_size, num_channels).
        if not config.get('use_sliding_windows', True):
            # If original was (batch, features_flat), and decoder outputs (batch, features_flat, 1)
            # or (batch, some_seq_len, some_channels) that needs to be reshaped back.
            # This requires knowing the 'original_feature_size' if it was initially flat.
            # The decoder output shape is (batch, output_shape_from_config, num_channels_from_config)
            # If original was (batch, N_features) and decoder outputs (batch, N_features, 1), then reshape.
            if 'original_feature_size' in config and decoded_data.shape[1] * decoded_data.shape[2] == config['original_feature_size']:
                 decoded_data = decoded_data.reshape((decoded_data.shape[0], config['original_feature_size']))
                 print(f"[decode_data] Reshaped decoded data to match original feature size: {decoded_data.shape}")
            elif decoded_data.shape[-1] == 1 and len(decoded_data.shape) ==3 : # (batch, features, 1) -> (batch, features)
                 decoded_data = np.squeeze(decoded_data, axis=-1)
                 print(f"[decode_data] Squeezed decoded data: {decoded_data.shape}")
            # else, the shape might be as intended or requires more specific handling.
        else: # Using sliding windows
            # The output is (batch, window_size, num_channels).
            # If it needs to be flattened back to (batch, window_size * num_channels) for comparison:
            # decoded_data = decoded_data.reshape((decoded_data.shape[0], -1))
            # print(f"[decode_data] Flattened decoded data for sliding window output: {decoded_data.shape}")
            print(f"[decode_data] Decoded data shape (sliding window): {decoded_data.shape}")


        return decoded_data






    def save_encoder(self, file_path):
        if self.encoder_model:
            self.encoder_model.save(file_path)
            print(f"[save_encoder] Encoder model saved to {file_path}")
        else:
            print("[save_encoder] Encoder model not available to save.")


    def save_decoder(self, file_path):
        if self.decoder_model:
            self.decoder_model.save(file_path)
            print(f"[save_decoder] Decoder model saved to {file_path}")
        else:
            print("[save_decoder] Decoder model not available to save.")


    def load_encoder(self, file_path):
        # For VAE encoder, compile=False is usually fine as it's part of the larger model or used for predict.
        self.encoder_model = load_model(file_path, compile=False)
        print(f"[load_encoder] Encoder model loaded from {file_path}")
        # Attempt to reconstruct shape_before_flatten_for_decoder if possible (already in plugin)
        if hasattr(self.encoder_plugin, 'load'): # if encoder plugin has its own load method
            self.encoder_plugin.load(file_path) # let plugin handle its internal state
        else: # Fallback for generic load
            try:
                # This assumes specific layer names, which might not be robust.
                # The encoder plugin should ideally handle restoring its state.
                conv_layer = self.encoder_model.get_layer("conv_output_before_flatten") # Example name
                self.encoder_plugin.shape_before_flatten_for_decoder = conv_layer.output_shape[1:]
            except ValueError:
                print("[load_encoder] Warning: Could not auto-detect 'shape_before_flatten_for_decoder' from loaded encoder. Ensure it's set if decoder configuration depends on it.")


    def load_decoder(self, file_path):
        self.decoder_model = load_model(file_path, compile=False)
        print(f"[load_decoder] Decoder model loaded from {file_path}")
        if hasattr(self.decoder_plugin, 'load'):
            self.decoder_plugin.load(file_path)


    def calculate_mse(self, original_data, reconstructed_data, config=None): # Added config for consistency
        print(f"[calculate_mse] Original data shape: {original_data.shape}")
        print(f"[calculate_mse] Reconstructed data shape: {reconstructed_data.shape}")

        # Ensure the data shapes match. This might require reshaping reconstructed_data
        # similar to how it's handled in decode_data or based on use_sliding_windows.
        # For a generic MSE, they must be broadcastable or identical.
        # Assuming original_data is the ground truth shape.
        if original_data.shape != reconstructed_data.shape:
            try:
                # Attempt a simple reshape if total elements match
                if np.prod(original_data.shape) == np.prod(reconstructed_data.shape):
                    reconstructed_data = reconstructed_data.reshape(original_data.shape)
                    print(f"[calculate_mse] Reshaped reconstructed data to: {reconstructed_data.shape}")
                else:
                    raise ValueError(f"Shape mismatch and element count mismatch: original {original_data.shape} vs reconstructed {reconstructed_data.shape}")
            except Exception as e:
                 raise ValueError(f"Shape mismatch: original data shape {original_data.shape} does not match reconstructed data shape {reconstructed_data.shape}. Error: {e}")


        # Calculate MSE
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"[calculate_mse] Calculated MSE: {mse}")
        return mse


    def calculate_mae(self, original_data, reconstructed_data, config=None): # Added config
        """
        Calculate the Mean Absolute Error (MAE) between original and reconstructed data.
        """
        print(f"[calculate_mae] Original data shape: {original_data.shape}")
        print(f"[calculate_mae] Reconstructed data shape: {reconstructed_data.shape}")

        # Ensure the data shapes match
        if original_data.shape != reconstructed_data.shape:
            try:
                if np.prod(original_data.shape) == np.prod(reconstructed_data.shape):
                    reconstructed_data = reconstructed_data.reshape(original_data.shape)
                    print(f"[calculate_mae] Reshaped reconstructed data to: {reconstructed_data.shape}")
                else:
                    raise ValueError(f"Shape mismatch and element count mismatch: original {original_data.shape} vs reconstructed {reconstructed_data.shape}")
            except Exception as e:
                raise ValueError(f"Shape mismatch: original data shape {original_data.shape} does not match reconstructed data shape {reconstructed_data.shape}. Error: {e}")

        # Calculate MAE consistently
        mae = tf.reduce_mean(tf.abs(tf.cast(original_data, tf.float32) - tf.cast(reconstructed_data, tf.float32))).numpy()
        print(f"[calculate_mae] Calculated MAE: {mae}")
        return mae


    # Removed duplicate calculate_mse method. The one above is kept.





