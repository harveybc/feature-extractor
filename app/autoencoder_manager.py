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

set_global_policy('mixed_float16')

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

            # Get the encoder's output shape
            encoder_output_shape = self.encoder_model.output_shape[1:]  # Exclude batch size
            print(f"Encoder output shape: {encoder_output_shape}")

            # Configure the decoder size, passing the encoder's output shape
            self.decoder_plugin.configure_size(interface_size, input_shape, num_channels, encoder_output_shape, use_sliding_windows, config)

            # Get the decoder model
            self.decoder_model = self.decoder_plugin.model
            print("[build_autoencoder] Decoder model built and compiled successfully")
            self.decoder_model.summary()

            # Build autoencoder model
            autoencoder_output = self.decoder_model(self.encoder_model.output)
            self.autoencoder_model = Model(inputs=self.encoder_model.input, outputs=autoencoder_output, name="autoencoder")
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
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                # Flatten inputs to ensure they are 2D.
                y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
                y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
                K_xx = gaussian_kernel_matrix(y_true, y_true, sigma)
                K_yy = gaussian_kernel_matrix(y_pred, y_pred, sigma)
                K_xy = gaussian_kernel_matrix(y_true, y_pred, sigma)
                # Use consistent dtype for statistical computations
                dtype = K_xx.dtype
                m = tf.cast(tf.shape(y_true)[0], dtype)
                n = tf.cast(tf.shape(y_pred)[0], dtype)
                # Compute the unbiased MMD statistic.
                mmd = tf.reduce_sum(K_xx) / (m * m) \
                      + tf.reduce_sum(K_yy) / (n * n) \
                      - tf.constant(2.0, dtype=dtype) * tf.reduce_sum(K_xy) / (m * n)
                return mmd

            # Combined loss: reconstruction (Huber) loss + weighted MMD loss.
            def combined_loss(y_true, y_pred):
                huber_loss = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
                sigma = config.get('mmd_sigma', 1.0)  # Configure kernel width if needed.
                stat_weight = config.get('statistical_loss_weight', 1.0)
                mmd = mmd_loss_term(y_true, y_pred, sigma)
                return huber_loss + stat_weight * mmd

            # Optional: Define a metric to monitor the MMD term during training.
            def mmd_metric(y_true, y_pred):
                sigma = config.get('mmd_sigma', 1.0)
                return mmd_loss_term(y_true, y_pred, sigma)
            # --- End Updated Loss Definition using MMD ---

            # Compile autoencoder with the combined loss and additional metric.
            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss=combined_loss,
                metrics=['mae', mmd_metric],
                run_eagerly=True
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
            self.calculate_dataset_information(data, config)

            print(f"[train_autoencoder] Training autoencoder with data shape: {data.shape}")

            # --- Setup Callbacks ---
            min_delta_early_stopping = config.get("min_delta", config.get("min_delta", 1e-4))
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
                # Removed: ClearMemoryCallback(), # <<< REMOVED THIS LINE
            ]
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
                    normalized_column = (column_data - min_val) / (max_val - min_val)  # Min-max normalization
                    normalized_columns.append(normalized_column)
            elif len(data.shape) == 2:  # Row-by-row data
                for col in range(data.shape[1]):
                    column_data = data[:, col]  # No need to flatten, already 1D
                    min_val, max_val = column_data.min(), column_data.max()
                    normalized_column = (column_data - min_val) / (max_val - min_val)  # Min-max normalization
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
                tf.greater(std_val, 0),
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
                sampling_frequency = tf.constant(0.0, dtype=tf.float32)

            # Calculate Shannon-Hartley channel capacity and total useful information
            channel_capacity = tf.cond(
                tf.math.logical_and(tf.greater(snr, 0), tf.greater(sampling_frequency, 0)),
                lambda: sampling_frequency * tf.math.log(1 + snr) / tf.math.log(2.0),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            total_information_bits = channel_capacity * num_samples * sampling_period_seconds

            # Calculate entropy using TensorFlow histogram binning
            bins = 1000  # Increased bin count for better precision
            histogram = tf.histogram_fixed_width(concatenated_data_tf, [0.0, 1.0], nbins=bins)
            histogram = tf.cast(histogram, tf.float32)
            probabilities = histogram / tf.reduce_sum(histogram)
            entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-10) / tf.math.log(2.0))  # Avoid log(0)

            # Log calculated information
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
            data = np.expand_dims(data, axis=-1)
            print(f"[evaluate] Reshaped {dataset_name} data for Conv1D compatibility: {data.shape}")

        # Evaluate the autoencoder
        results = self.autoencoder_model.evaluate(data, data, verbose=1)
        mse, mae = results[0], results[1]  # Retrieve MSE (loss) and MAE

        print(f"[evaluate] {dataset_name} Evaluation results - MSE: {mse}, MAE: {mae}")
        return mse, mae






    def encode_data(self, data, config):
        print(f"[encode_data] Encoding data with shape: {data.shape}")

        # Determine if sliding windows are used
        use_sliding_windows = config.get('use_sliding_windows', True)

        # Reshape data for sliding windows or row-by-row processing
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
            # For row-by-row data, add a channel dimension
            num_channels = 1
            data = np.expand_dims(data, axis=-1)

        print(f"[encode_data] Reshaped data shape for encoding: {data.shape}")

        # Perform encoding
        try:
            encoded_data = self.encoder_model.predict(data)
            print(f"[encode_data] Encoded data shape: {encoded_data.shape}")
            return encoded_data
        except Exception as e:
            print(f"[encode_data] Exception occurred during encoding: {e}")
            raise ValueError("[encode_data] Failed to encode data. Please check model compatibility and data shape.")




   
    def decode_data(self, encoded_data, config):
        print(f"[decode_data] Decoding data with shape: {encoded_data.shape}")
        decoded_data = self.decoder_model.predict(encoded_data)

        if not config['use_sliding_windows']:
            # Reshape decoded data for row-by-row processing
            decoded_data = decoded_data.reshape((decoded_data.shape[0], config['original_feature_size']))
            print(f"[decode_data] Reshaped decoded data to match original feature size: {decoded_data.shape}")
        else:
            print(f"[decode_data] Decoded data shape: {decoded_data.shape}")

        return decoded_data






    def save_encoder(self, file_path):
        self.encoder_model.save(file_path)
        print(f"[save_encoder] Encoder model saved to {file_path}")

    def save_decoder(self, file_path):
        self.decoder_model.save(file_path)
        print(f"[save_decoder] Decoder model saved to {file_path}")

    def load_encoder(self, file_path):
        self.encoder_model = load_model(file_path)
        print(f"[load_encoder] Encoder model loaded from {file_path}")

    def load_decoder(self, file_path):
        self.decoder_model = load_model(file_path)
        print(f"[load_decoder] Decoder model loaded from {file_path}")

    def calculate_mse(self, original_data, reconstructed_data, config):
        print(f"[calculate_mse] Original data shape: {original_data.shape}")
        print(f"[calculate_mse] Reconstructed data shape: {reconstructed_data.shape}")

        use_sliding_windows = config.get('use_sliding_windows', True)

        # Handle sliding windows: Aggregate reconstructed data if necessary
        if use_sliding_windows:
            window_size = config['window_size']
            num_channels = original_data.shape[1] // window_size

            # Reshape reconstructed data to match original sliding window format
            if reconstructed_data.shape != original_data.shape:
                print("[calculate_mse] Adjusting reconstructed data shape for sliding window comparison...")
                reconstructed_data = reconstructed_data.reshape(
                    (original_data.shape[0], original_data.shape[1])
                )

        # Ensure the data shapes match after adjustments
        if original_data.shape != reconstructed_data.shape:
            raise ValueError(f"Shape mismatch: original data shape {original_data.shape} does not match reconstructed data shape {reconstructed_data.shape}")

        # Calculate MSE
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"[calculate_mse] Calculated MSE: {mse}")
        return mse


    def calculate_mae(self, original_data, reconstructed_data, config):
        """
        Calculate the Mean Absolute Error (MAE) between original and reconstructed data.
        """
        print(f"[calculate_mae] Original data shape: {original_data.shape}")
        print(f"[calculate_mae] Reconstructed data shape: {reconstructed_data.shape}")

        # Ensure the data shapes match
        if original_data.shape != reconstructed_data.shape:
            raise ValueError(f"Shape mismatch: original data shape {original_data.shape} does not match reconstructed data shape {reconstructed_data.shape}")

        # Calculate MAE consistently
        mae = tf.reduce_mean(tf.abs(original_data - reconstructed_data)).numpy()
        print(f"[calculate_mae] Calculated MAE: {mae}")
        return mae


    def calculate_mse(self, original_data, reconstructed_data, config):
        """
        Calculate the Mean Squared Error (MSE) between original and reconstructed data.
        """
        print(f"[calculate_mse] Original data shape: {original_data.shape}")
        print(f"[calculate_mse] Reconstructed data shape: {reconstructed_data.shape}")

        # Ensure the data shapes match
        if original_data.shape != reconstructed_data.shape:
            raise ValueError(f"Shape mismatch: original data shape {original_data.shape} does not match reconstructed data shape {reconstructed_data.shape}")

        # Calculate MSE consistently
        mse = tf.reduce_mean(tf.square(original_data - reconstructed_data)).numpy()
        print(f"[calculate_mse] Calculated MSE: {mse}")
        return mse





