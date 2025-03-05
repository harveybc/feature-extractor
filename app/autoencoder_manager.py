import gc  # For garbage collection
import numpy as np
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.mixed_precision import set_global_policy

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Error setting GPU memory growth:", e)

# Set global mixed precision policy
set_global_policy('mixed_float16')

class UpdateOverfitPenalty(Callback):
    """
    Custom Callback to update the overfit penalty value used in the loss function.
    At the end of each epoch, it computes the difference between validation MAE and training MAE,
    multiplies it by a scaling factor (0.1), and updates a TensorFlow variable in the model.
    The penalty is applied only if validation MAE is higher than training MAE.
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_mae = logs.get('mae')
        val_mae = logs.get('val_mae')
        if train_mae is None or val_mae is None:
            print("[UpdateOverfitPenalty] MAE metrics not available; overfit penalty not updated.")
            return
        penalty = 0.1 * max(0, val_mae - train_mae)
        tf.keras.backend.set_value(self.model.overfit_penalty, penalty)
        print(f"[UpdateOverfitPenalty] Epoch {epoch+1}: Updated overfit penalty to {penalty:.6f}")

class DebugLearningRateCallback(Callback):
    """
    Debug Callback that prints the current learning rate,
    the wait counter for EarlyStopping, and for the LR reducer.
    """
    def __init__(self, early_stopping_cb, lr_reducer_cb):
        super(DebugLearningRateCallback, self).__init__()
        self.early_stopping_cb = early_stopping_cb
        self.lr_reducer_cb = lr_reducer_cb

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        es_wait = getattr(self.early_stopping_cb, "wait", None)
        lr_wait = getattr(self.lr_reducer_cb, "wait", None)
        best_val = getattr(self.lr_reducer_cb, "best", None)
        print(f"[DebugLR] Epoch {epoch+1}: Learning Rate = {current_lr:.6e}, EarlyStopping wait = {es_wait}, LRReducer wait = {lr_wait}, LRReducer best = {best_val}")

class MemoryCleanupCallback(Callback):
    """
    Callback to force garbage collection at the end of each epoch.
    This can help free up unused memory and mitigate memory leaks.
    """
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        #print(f"[MemoryCleanup] Epoch {epoch+1}: Garbage collection executed.")

def gaussian_kernel_sum(x, y, sigma, chunk_size=16):
    """
    Compute the sum of Gaussian kernel values between each pair of rows in x and y
    in a memory-efficient manner by processing in chunks.
    x: Tensor of shape [n, d]
    y: Tensor of shape [m, d]
    sigma: Bandwidth parameter for the Gaussian kernel.
    Returns a scalar equal to the sum of exp(-||x_i - y_j||^2/(2*sigma^2)) for all pairs.
    """
    n = tf.shape(x)[0]
    total = tf.constant(0.0, dtype=tf.float32)
    i = tf.constant(0)
    
    def cond(i, total):
        return tf.less(i, n)
    
    def body(i, total):
        end_i = tf.minimum(i + chunk_size, n)
        x_chunk = x[i:end_i]  # shape [chunk, d]
        diff = tf.expand_dims(x_chunk, axis=1) - tf.expand_dims(y, axis=0)  # shape [chunk, m, d]
        squared_diff = tf.reduce_sum(tf.square(diff), axis=2)  # shape [chunk, m]
        # Use tf.square for sigma to avoid potential issues with inline exponentiation.
        divisor = 2.0 * tf.square(sigma)
        kernel_chunk = tf.exp(-squared_diff / divisor)
        total += tf.reduce_sum(kernel_chunk)
        return i + chunk_size, total
    
    i, total = tf.while_loop(cond, body, [i, total])
    return total

def mmd_loss_term(y_true, y_pred, sigma, chunk_size=16):
    """
    Compute the Maximum Mean Discrepancy (MMD) loss between y_true and y_pred using
    a memory-efficient chunked Gaussian kernel sum.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    sum_K_xx = gaussian_kernel_sum(y_true, y_true, sigma, chunk_size)
    sum_K_yy = gaussian_kernel_sum(y_pred, y_pred, sigma, chunk_size)
    sum_K_xy = gaussian_kernel_sum(y_true, y_pred, sigma, chunk_size)
    m = tf.cast(tf.shape(y_true)[0], tf.float32)
    n = tf.cast(tf.shape(y_pred)[0], tf.float32)
    mmd = sum_K_xx / (m * m) + sum_K_yy / (n * n) - 2 * sum_K_xy / (m * n)
    return mmd

def mmd_metric(y_true, y_pred, config):
    sigma = config.get('mmd_sigma', 1.0)
    return mmd_loss_term(y_true, y_pred, sigma, chunk_size=16)

class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin):
        """
        Initialize the AutoencoderManager with provided encoder and decoder plugins.
        """
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.autoencoder_model = None
        self.encoder_model = None
        self.decoder_model = None
        self.overfit_penalty = None  
        print(f"[AutoencoderManager] Initialized with encoder plugin and decoder plugin")

    def build_autoencoder(self, input_shape, interface_size, config, num_channels):
        """
        Build and compile the autoencoder by configuring the encoder and decoder.
        """
        try:
            print("[build_autoencoder] Starting to build autoencoder...")
            use_sliding_windows = config.get('use_sliding_windows', True)
            
            self.encoder_plugin.configure_size(input_shape, interface_size, num_channels, use_sliding_windows)
            self.encoder_model = self.encoder_plugin.encoder_model
            print("[build_autoencoder] Encoder model built and compiled successfully")
            self.encoder_model.summary()
            
            encoder_preflatten = self.encoder_plugin.pre_flatten_shape
            encoder_skips = self.encoder_plugin.skip_connections
            print(f"Encoder pre-flatten shape: {encoder_preflatten}")
            
            self.decoder_plugin.configure_size(interface_size, input_shape, num_channels,
                                               encoder_preflatten, use_sliding_windows, encoder_skips)
            self.decoder_model = self.decoder_plugin.model
            print("[build_autoencoder] Decoder model built and compiled successfully")
            self.decoder_model.summary()
            
            latent = self.encoder_model.output
            decoder_inputs = [latent] + encoder_skips
            autoencoder_output = self.decoder_model(decoder_inputs)
            self.autoencoder_model = Model(inputs=self.encoder_model.input, outputs=autoencoder_output, name="autoencoder")
            
            self.overfit_penalty = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            self.autoencoder_model.overfit_penalty = self.overfit_penalty
            
            initial_lr = config.get('learning_rate', 0.01)
            adam_optimizer = Adam(
                learning_rate=initial_lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                amsgrad=False,
                clipnorm=1.0,
                clipvalue=0.5
            )
            
            def combined_loss(y_true, y_pred):
                huber_loss = Huber(delta=1.0)(y_true, y_pred)
                sigma = config.get('mmd_sigma', 1.0)
                stat_weight = config.get('statistical_loss_weight', 1.0)
                mmd = mmd_loss_term(y_true, y_pred, sigma, chunk_size=16)
                penalty_term = tf.cast(1.0, tf.float32) * tf.stop_gradient(self.overfit_penalty)
                return huber_loss + (stat_weight * mmd) + penalty_term
            
            if config.get('use_mmd', False):
                loss_fn = combined_loss
                metrics = ['mae', lambda yt, yp: mmd_metric(yt, yp, config)]
            else:
                loss_fn = Huber(delta=1.0)
                metrics = ['mae']
            
            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss=loss_fn,
                metrics=metrics
            )
            print("[build_autoencoder] Autoencoder model built and compiled successfully")
            self.autoencoder_model.summary()
        except Exception as e:
            print(f"[build_autoencoder] Exception occurred: {e}")
            raise

    def train_autoencoder(self, data, val_data, epochs=100, batch_size=128, config=None):
        """
        Train the autoencoder using provided training and validation data.
        The UpdateOverfitPenalty callback updates the penalty term at the end of each epoch.
        ReduceLROnPlateau monitors the validation MAE and reduces the learning rate if no improvement is observed.
        DebugLearningRateCallback reports the current learning rate and patience counters.
        MemoryCleanupCallback forces garbage collection at the end of each epoch.
        """
        try:
            print(f"[train_autoencoder] Received data with shape: {data.shape}")
            use_sliding_windows = config.get('use_sliding_windows', True)
            if not use_sliding_windows and len(data.shape) == 2:
                print("[train_autoencoder] Reshaping data to add channel dimension for Conv1D compatibility...")
                data = np.expand_dims(data, axis=-1)
                print(f"[train_autoencoder] Reshaped data shape: {data.shape}")
            
            num_channels = data.shape[-1]
            input_shape = data.shape[1]
            interface_size = self.encoder_plugin.params.get('interface_size', 4)
            
            if not self.autoencoder_model:
                self.build_autoencoder(input_shape, interface_size, config, num_channels)
            
            if np.isnan(data).any():
                raise ValueError("[train_autoencoder] Training data contains NaN values. Please check your preprocessing pipeline.")
            
            self.calculate_dataset_information(data, config)
            print(f"[train_autoencoder] Training autoencoder with data shape: {data.shape}")
            
            early_patience = config.get('early_patience', 10)
            early_monitor = config.get('early_monitor', 'val_loss')
            early_stopping = EarlyStopping(monitor=early_monitor, patience=early_patience, restore_best_weights=True)
            
            update_penalty_cb = UpdateOverfitPenalty()
            
            lr_reducer = ReduceLROnPlateau(
                monitor=early_monitor,
                factor=0.1,
                patience=int(early_patience / 2),
                verbose=1,
                min_lr=config.get('min_lr', 1e-6)
            )
            
            debug_lr_cb = DebugLearningRateCallback(early_stopping, lr_reducer)
            memory_cleanup_cb = MemoryCleanupCallback()
            
            if val_data is not None and isinstance(val_data, np.ndarray):
                val_data = (val_data, val_data)
            
            history = self.autoencoder_model.fit(
                data,
                data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[early_stopping, update_penalty_cb, lr_reducer, debug_lr_cb, memory_cleanup_cb],
                validation_data=val_data
            )
            
            print(f"[train_autoencoder] Training loss values: {history.history['loss']}")
            print("[train_autoencoder] Training completed.")
        except Exception as e:
            print(f"[train_autoencoder] Exception occurred during training: {e}")
            raise

    def calculate_dataset_information(self, data, config):
        """
        Calculate dataset statistics including entropy and channel capacity using the Shannon-Hartley theorem.
        Supports both 2D (row-by-row) and 3D (sliding windows) data.
        """
        try:
            print("[calculate_dataset_information] Calculating dataset entropy and useful information...")
            normalized_columns = []
            if len(data.shape) == 3:
                for col in range(data.shape[-1]):
                    column_data = data[:, :, col].flatten()
                    min_val, max_val = column_data.min(), column_data.max()
                    normalized_column = (column_data - min_val) / (max_val - min_val)
                    normalized_columns.append(normalized_column)
            elif len(data.shape) == 2:
                for col in range(data.shape[1]):
                    column_data = data[:, col]
                    min_val, max_val = column_data.min(), column_data.max()
                    normalized_column = (column_data - min_val) / (max_val - min_val)
                    normalized_columns.append(normalized_column)
            else:
                raise ValueError("[calculate_dataset_information] Unsupported data shape for processing.")
            
            concatenated_data = np.concatenate(normalized_columns, axis=0)
            num_samples = concatenated_data.shape[0]
            concatenated_data_tf = tf.convert_to_tensor(concatenated_data, dtype=tf.float32)
            mean_val = tf.reduce_mean(concatenated_data_tf)
            std_val = tf.math.reduce_std(concatenated_data_tf)
            snr = tf.cond(
                tf.greater(std_val, 0),
                lambda: (mean_val / std_val) ** 2,
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            
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
                sampling_frequency = tf.constant(0.0, dtype=tf.float32)
            
            channel_capacity = tf.cond(
                tf.math.logical_and(tf.greater(snr, 0), tf.greater(sampling_frequency, 0)),
                lambda: sampling_frequency * tf.math.log(1 + snr) / tf.math.log(2.0),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            total_information_bits = channel_capacity * num_samples * sampling_period_seconds
            
            bins = 1000
            histogram = tf.histogram_fixed_width(concatenated_data_tf, [0.0, 1.0], nbins=bins)
            histogram = tf.cast(histogram, tf.float32)
            probabilities = histogram / tf.reduce_sum(histogram)
            entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-10) / tf.math.log(2.0))
            
            print(f"[calculate_dataset_information] Calculated SNR: {snr.numpy()}")
            print(f"[calculate_dataset_information] Sampling frequency: {sampling_frequency.numpy()} Hz")
            print(f"[calculate_dataset_information] Channel capacity: {channel_capacity.numpy()} bits/second")
            print(f"[calculate_dataset_information] Total useful information: {total_information_bits.numpy()} bits")
            print(f"[calculate_dataset_information] Entropy: {entropy.numpy()} bits")
        except Exception as e:
            print(f"[calculate_dataset_information] Exception occurred: {e}")
            raise

    def evaluate(self, data, dataset_name, config):
        """
        Evaluate the autoencoder on the provided dataset by computing MSE, MAE, and R².
        """
        print(f"[evaluate] Evaluating {dataset_name} data with shape: {data.shape}")
        if not config.get('use_sliding_windows', True) and len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
            print(f"[evaluate] Reshaped {dataset_name} data for Conv1D compatibility: {data.shape}")
        results = self.autoencoder_model.evaluate(data, data, verbose=1)
        mse, mae = results[0], results[1]
        predictions = self.autoencoder_model.predict(data)
        r2 = self.calculate_r2(data, predictions)
        print(f"[evaluate] {dataset_name} Evaluation results - MSE: {mse}, MAE: {mae}, R²: {r2}")
        return mse, mae, r2

    def encode_data(self, data, config):
        """
        Encode input data using the encoder model.
        """
        print(f"[encode_data] Encoding data with shape: {data.shape}")
        use_sliding_windows = config.get('use_sliding_windows', True)
        if use_sliding_windows:
            if config['window_size'] > data.shape[1]:
                raise ValueError("[encode_data] window_size cannot be greater than the number of features in the data.")
            num_channels = data.shape[1] // config['window_size']
            if num_channels <= 0:
                raise ValueError("[encode_data] Invalid num_channels calculated for sliding window data.")
            if data.shape[1] % config['window_size'] != 0:
                raise ValueError("[encode_data] data.shape[1] must be divisible by window_size for sliding windows.")
            data = data.reshape((data.shape[0], config['window_size'], num_channels))
        else:
            num_channels = 1
            data = np.expand_dims(data, axis=-1)
        print(f"[encode_data] Reshaped data for encoding: {data.shape}")
        try:
            encoded_data = self.encoder_model.predict(data)
            print(f"[encode_data] Encoded data shape: {encoded_data.shape}")
            return encoded_data
        except Exception as e:
            print(f"[encode_data] Exception during encoding: {e}")
            raise ValueError("[encode_data] Failed to encode data. Check model compatibility and data shape.")

    def decode_data(self, encoded_data, config):
        """
        Decode latent representations using the decoder model.
        """
        print(f"[decode_data] Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.decoder_model.predict(encoded_data)
        if not config['use_sliding_windows']:
            decoded_data = decoded_data.reshape((decoded_data.shape[0], config['original_feature_size']))
            print(f"[decode_data] Reshaped decoded data to match original feature size: {decoded_data.shape}")
        else:
            print(f"[decode_data] Decoded data shape: {decoded_data.shape}")
        return decoded_data

    def save_encoder(self, file_path):
        """Save the encoder model."""
        self.encoder_model.save(file_path)
        print(f"[save_encoder] Encoder model saved to {file_path}")

    def save_decoder(self, file_path):
        """Save the decoder model."""
        self.decoder_model.save(file_path)
        print(f"[save_decoder] Decoder model saved to {file_path}")

    def load_encoder(self, file_path):
        """Load the encoder model."""
        self.encoder_model = load_model(file_path)
        print(f"[load_encoder] Encoder model loaded from {file_path}")

    def load_decoder(self, file_path):
        """Load the decoder model."""
        self.decoder_model = load_model(file_path)
        print(f"[load_decoder] Decoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data, config):
        """
        Calculate Mean Squared Error (MSE) between original and reconstructed data.
        """
        print(f"[calculate_mse] Original data shape: {original_data.shape}")
        print(f"[calculate_mse] Reconstructed data shape: {reconstructed_data.shape}")
        if original_data.shape != reconstructed_data.shape:
            raise ValueError(f"Shape mismatch: original {original_data.shape} vs reconstructed {reconstructed_data.shape}")
        mse = tf.reduce_mean(tf.square(original_data - reconstructed_data)).numpy()
        print(f"[calculate_mse] Calculated MSE: {mse}")
        return mse

    def calculate_mae(self, original_data, reconstructed_data, config):
        """
        Calculate Mean Absolute Error (MAE) between original and reconstructed data.
        """
        print(f"[calculate_mae] Original data shape: {original_data.shape}")
        print(f"[calculate_mae] Reconstructed data shape: {reconstructed_data.shape}")
        if original_data.shape != reconstructed_data.shape:
            raise ValueError(f"Shape mismatch: original {original_data.shape} vs reconstructed {reconstructed_data.shape}")
        mae = tf.reduce_mean(tf.abs(original_data - reconstructed_data)).numpy()
        print(f"[calculate_mae] Calculated MAE: {mae}")
        return mae

    def calculate_r2(self, y_true, y_pred):
        """
        Calculate the R² (Coefficient of Determination) score.
        """
        print(f"Calculating R² for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=1, keepdims=True)) ** 2, axis=1)
        r2_scores = 1 - (ss_res / ss_tot)
        r2_scores = np.where(ss_tot == 0, 0, r2_scores)
        r2 = np.mean(r2_scores)
        print(f"Calculated R²: {r2}")
        return r2
