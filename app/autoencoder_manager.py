import numpy as np
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import backend as K
from keras.callbacks import LambdaCallback, ReduceLROnPlateau
from keras.layers import Input, Dense, Flatten, Reshape, Dropout, Lambda
from keras.regularizers import l2
from keras.layers import BatchNormalization, LeakyReLU
from keras.layers import MaxPooling1D, UpSampling1D

#set_global_policy('mixed_float16')

#define tensorflow global variable mmd_total as a float
mmd_total = tf.Variable(0.0, dtype=tf.float32, trainable=False)

# ============================================================================
# Callback to dynamically adjust mmd_weight so Huber and MMD remain same order
# ============================================================================
class MMDWeightAdjustmentCallback(tf.keras.callbacks.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def on_epoch_end(self, epoch, logs=None):
        # compute current Huber and MMD contributions
        huber_val = logs['loss'] - logs.get('mmd_metric', 0.0)
        mmd_val   = logs.get('mmd_metric', 0.0)
        ratio     = huber_val / (mmd_val + 1e-12)
        # keep ratio roughly in [0.5, 2.0]
        if ratio > 2.0:
            self.cfg['mmd_weight'] *= 1.1
        elif ratio < 0.5:
            self.cfg['mmd_weight'] *= 0.9
        print(f"[MMDWeightAdjust] Epoch {epoch+1}: ratio={ratio:.3f}, new mmd_weight={self.cfg['mmd_weight']:.5f}")

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


class Sampling(keras.layers.Layer): # Changed to keras.layers.Layer
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        config = super().get_config()
        return config

# Custom VAE Model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, kl_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs): # Defines the forward pass for inference
        z_mean, z_log_var = self.encoder(inputs)
        _z_mean, _z_log_var = self.encoder(inputs)
        _z = Sampling()([_z_mean, _z_log_var]) # Apply sampling
        reconstruction = self.decoder(_z)
        return reconstruction


    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0] # Assuming data is (x, y) or just x
        else:
            x = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x)
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            reconstruction = self.decoder(z) # Decoder takes the sampled z

            # Reconstruction loss (e.g., Huber)
            # self.compiled_loss is the one provided in compile() method (e.g. Huber)
            reconstruction_loss = self.compiled_loss(x, reconstruction, regularization_losses=self.losses)
            
            # KL divergence
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
            )
            kl_loss = tf.reduce_mean(kl_loss)
            
            total_loss = reconstruction_loss + self.kl_weight * kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            # Include other metrics from self.compiled_metrics
            **{m.name: m.result() for m in self.metrics if m.name not in ["total_loss", "reconstruction_loss", "kl_loss"]}
        }


class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin):
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.autoencoder_model = None # This will be the VAE model (instance of VAE class)
        self.encoder_model = None     # Keras Model for encoder part
        self.decoder_model = None     # Keras Model for decoder part
        # self.kl_loss_tracker and self.reconstruction_loss_tracker are now part of the VAE class
        print(f"[AutoencoderManager VAE] Initialized with encoder plugin and decoder plugin")

    def build_autoencoder(self, input_shape, interface_size, config, num_channels):
        try:
            print("[build_autoencoder VAE] Starting to build VAE...")
            use_sliding_windows = config.get('use_sliding_windows', True)

            # 1. Configure and get the Encoder Model (outputs z_mean, z_log_var)
            self.encoder_plugin.configure_size(input_shape, interface_size, num_channels, use_sliding_windows, config)
            self.encoder_model = self.encoder_plugin.encoder_model # This is a Keras Model
            if not self.encoder_model:
                raise ValueError("VAE Encoder model not built by plugin.")
            print("[build_autoencoder VAE] VAE Encoder part (outputs [z_mean, z_log_var]) built.")

            # 2. Configure and get the Decoder Model (takes sampled z as input)
            encoder_shape_before_flatten = self.encoder_plugin.shape_before_flatten_for_decoder
            if encoder_shape_before_flatten is None or None in encoder_shape_before_flatten:
                raise ValueError("encoder_plugin.shape_before_flatten_for_decoder is not set correctly or contains None.")
            
            self.decoder_plugin.configure_size(interface_size, input_shape, num_channels, encoder_shape_before_flatten, use_sliding_windows, config)
            self.decoder_model = self.decoder_plugin.model # This is a Keras Model
            if not self.decoder_model:
                raise ValueError("VAE Decoder model not built by plugin.")
            print("[build_autoencoder VAE] VAE Decoder part (takes sampled z) built.")

            # 3. Instantiate the custom VAE model
            self.autoencoder_model = VAE(
                encoder=self.encoder_model, 
                decoder=self.decoder_model,
                kl_weight=config.get('kl_weight', 1.0),
                name="vae_custom_model"
            )

            # 4. Compile the VAE model
            adam_optimizer = Adam(
                learning_rate=config.get('learning_rate', 0.001),
                beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
            )
            reconstruction_loss_fn = Huber(delta=config.get('huber_delta', 1.0))
            
            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss=reconstruction_loss_fn, # This is for y_true vs y_pred (reconstruction)
                metrics=['mae'] # Add other reconstruction metrics if needed
            )

            print("[build_autoencoder VAE] Custom VAE model built and compiled successfully.")
            # To see the structure, you might need to call build on the VAE model first
            # self.autoencoder_model.build(input_shape=(None, input_shape, num_channels)) # Or appropriate input spec
            # self.autoencoder_model.summary() # Then summary
        except Exception as e:
            print(f"[build_autoencoder VAE] Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            raise
    # ... rest of AutoencoderManager, ensure train_autoencoder calls self.autoencoder_model.fit ...
    # encode_data should use self.encoder_model.predict()
    # decode_data should use self.decoder_model.predict(sampled_z)
    # save/load_full_vae will save/load self.autoencoder_model (the VAE instance)
    # save/load_encoder will save/load self.encoder_model (the Keras Model part)
    # save/load_decoder will save/load self.decoder_model (the Keras Model part)

    def save_full_vae(self, file_path):
        if self.autoencoder_model:
            # Saving a custom model might require custom object handling if it contains
            # layers not automatically recognized, but VAE itself is a Model subclass.
            # The internal encoder/decoder are standard Keras models.
            self.autoencoder_model.save(file_path) # Keras 3 should handle this better
            print(f"[save_full_vae] Full VAE model saved to {file_path}")
        else:
            print("Full VAE model not available to save.")

    def load_full_vae(self, file_path):
        # When loading, provide custom objects if VAE class or Sampling is not registered
        custom_objects = {'VAE': VAE, 'Sampling': Sampling}
        self.autoencoder_model = keras.models.load_model(file_path, custom_objects=custom_objects)
        print(f"[load_full_vae] Full VAE model loaded from {file_path}")
        # Re-link internal encoder/decoder if necessary for other methods
        self.encoder_model = self.autoencoder_model.encoder
        self.decoder_model = self.autoencoder_model.decoder
        self.encoder_plugin.encoder_model = self.encoder_model
        self.decoder_plugin.model = self.decoder_model
        # Potentially re-init plugin params from loaded models
        # self.encoder_plugin.load(None) # This might need adjustment
        print("[load_full_vae] References to VAE encoder and decoder parts updated.")

    def train_autoencoder(self, data, epochs=100, batch_size=128, config=None):
        try:
            print(f"[train_autoencoder VAE] Received data with shape: {data.shape}")
            use_sliding_windows = config.get('use_sliding_windows', True)

            if not use_sliding_windows and len(data.shape) == 2:
                data = np.expand_dims(data, axis=-1)
            
            num_channels = data.shape[-1]
            # input_shape for configure_size is sequence length (e.g., window_size)
            input_seq_len = data.shape[1] 
            interface_size = config.get('interface_size', 16) # Latent dimension

            if not self.autoencoder_model: # Build VAE if not already built
                self.build_autoencoder(input_seq_len, interface_size, config, num_channels)

            if np.isnan(data).any():
                raise ValueError("[train_autoencoder VAE] Training data contains NaN values.")

            # ... (keep calculate_dataset_information if still relevant) ...
            print(f"[train_autoencoder VAE] Training VAE with data shape: {data.shape}")
            
            # --- Setup Callbacks (can reuse your existing ones) ---
            # ... (your EarlyStopping, ReduceLROnPlateau, LambdaCallback setup) ...
            callbacks = [
                # Your existing callbacks
            ]

            # For VAE, y is the same as x for reconstruction loss
            history = self.autoencoder_model.fit(
                data, # x
                data, # y (for reconstruction loss)
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=callbacks,
                validation_split=config.get('validation_split', 0.2)
            )
            print(f"[train_autoencoder VAE] Training loss values: {history.history['loss']}")
            if 'val_loss' in history.history:
                 print(f"[train_autoencoder VAE] Validation loss values: {history.history['val_loss']}")
            print("[train_autoencoder VAE] Training completed.")
        except Exception as e:
            print(f"[train_autoencoder VAE] Exception occurred during training: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def encode_data(self, data, config=None): # config might not be needed if model is built
        if not self.encoder_model:
            raise ValueError("VAE Encoder model not available. Build or load first.")
        print(f"[encode_data VAE] Encoding data with shape: {data.shape}")
        # ... (data reshaping logic if needed, similar to your existing encode_data) ...
        # The VAE encoder model directly outputs [z_mean, z_log_var]
        encoded_outputs = self.encoder_model.predict(data)
        # encoded_outputs is a list: [z_mean_array, z_log_var_array]
        print(f"[encode_data VAE] Encoded data: z_mean shape {encoded_outputs[0].shape}, z_log_var shape {encoded_outputs[1].shape}")
        return encoded_outputs # Return the list

    def decode_data(self, encoded_z_sample, config=None): # Takes sampled z
        if not self.decoder_model:
            raise ValueError("VAE Decoder model not available. Build or load first.")
        # encoded_z_sample is the actual sampled latent vector 'z'
        print(f"[decode_data VAE] Decoding data (sampled z) with shape: {encoded_z_sample.shape}")
        decoded_data = self.decoder_model.predict(encoded_z_sample)
        # ... (reshaping logic for decoded_data if needed, similar to your existing decode_data) ...
        print(f"[decode_data VAE] Decoded data shape: {decoded_data.shape}")
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





