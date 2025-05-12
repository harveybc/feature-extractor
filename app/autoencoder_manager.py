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


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self): # Important for model saving/loading
        config = super().get_config()
        return config

class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin):
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.autoencoder_model = None # This will be the VAE model
        self.encoder_model = None     # This will be the model outputting [z_mean, z_log_var]
        self.decoder_model = None     # This will be the model taking sampled z as input
        # self.model = None # Redundant if self.autoencoder_model is used as the main model
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss") # For tracking KL loss
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss") # For tracking reconstruction loss
        print(f"[AutoencoderManager VAE] Initialized with encoder plugin and decoder plugin")

    def build_autoencoder(self, input_shape, interface_size, config, num_channels):
        try:
            print("[build_autoencoder VAE] Starting to build VAE...")
            use_sliding_windows = config.get('use_sliding_windows', True)

            # 1. Configure and get the VAE Encoder (outputs z_mean, z_log_var)
            self.encoder_plugin.configure_size(input_shape, interface_size, num_channels, use_sliding_windows, config)
            self.encoder_model = self.encoder_plugin.encoder_model # This model outputs [z_mean, z_log_var]
            if not self.encoder_model:
                raise ValueError("VAE Encoder model not built by plugin.")
            print("[build_autoencoder VAE] VAE Encoder model (outputs [z_mean, z_log_var]) built.")
            self.encoder_model.summary()

            # Get z_mean and z_log_var from the encoder's output list
            # The encoder_model.output is already a list [z_mean_tensor, z_log_var_tensor]
            z_mean_tensor, z_log_var_tensor = self.encoder_model.output 

            # 2. Create the Sampling layer
            # This layer takes [z_mean, z_log_var] and outputs sampled z
            z_sampling_layer = Sampling(name="vae_sampling_layer")([z_mean_tensor, z_log_var_tensor])

            # 3. Configure and get the VAE Decoder (takes sampled z as input)
            # We need `encoder_shape_before_flatten` from the encoder plugin
            encoder_shape_before_flatten = self.encoder_plugin.shape_before_flatten_for_decoder
            if encoder_shape_before_flatten is None:
                raise ValueError("encoder_plugin.shape_before_flatten_for_decoder is not set. Ensure encoder configures this.")
            
            self.decoder_plugin.configure_size(interface_size, input_shape, num_channels, encoder_shape_before_flatten, use_sliding_windows, config)
            self.decoder_model = self.decoder_plugin.model # This model takes sampled z
            if not self.decoder_model:
                raise ValueError("VAE Decoder model not built by plugin.")
            print("[build_autoencoder VAE] VAE Decoder model (takes sampled z) built.")
            self.decoder_model.summary()

            # 4. Connect components to form the VAE
            # The VAE output is the decoder's output when fed the sampled z
            vae_output = self.decoder_model(z_sampling_layer)
            
            # The VAE model takes the original encoder input and outputs the reconstruction
            self.autoencoder_model = Model(inputs=self.encoder_model.input, outputs=vae_output, name="vae_full")

            # Define optimizer (can reuse your existing Adam setup)
            adam_optimizer = Adam(
                learning_rate=config.get('learning_rate', 0.001),
                beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
            )

            # --- VAE Loss Function ---
            # Reconstruction loss (e.g., Huber, MSE, or your MMD on reconstruction)
            reconstruction_loss_fn = tf.keras.losses.Huber(delta=config.get('huber_delta', 1.0))
            # Or, if you want to keep your MMD for reconstruction:
            # def mmd_reconstruction_loss(y_true_recon, y_pred_recon):
            #     sigma = config.get('mmd_sigma_recon', 1.0) # Use a different sigma if needed
            #     return mmd_loss_term(y_true_recon, y_pred_recon, sigma) # Your existing mmd_loss_term
            # reconstruction_loss_fn = mmd_reconstruction_loss

            # KL divergence loss
            # z_mean_tensor and z_log_var_tensor are available from encoder_model.output
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var_tensor - tf.square(z_mean_tensor) - tf.exp(z_log_var_tensor), axis=-1
            )
            kl_loss = tf.reduce_mean(kl_loss) # Average over batch

            # Add KL loss to the VAE model's losses
            # This is a common way to add unconditional losses that depend on intermediate layers
            self.autoencoder_model.add_loss(config.get('kl_weight', 1.0) * kl_loss)
            
            # Compile VAE model
            # The 'loss' argument here will be the reconstruction loss.
            # The KL loss is added via `add_loss`.
            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss=reconstruction_loss_fn, # This is for y_true vs y_pred (reconstruction)
                metrics=['mae'] # Add other reconstruction metrics if needed
                # run_eagerly=True # For debugging, then set to False
            )
            # To track KL loss as a metric (optional, as it's part of total loss)
            # self.autoencoder_model.add_metric(kl_loss, name='kl_divergence_metric')


            print("[build_autoencoder VAE] VAE model built and compiled successfully.")
            self.autoencoder_model.summary()
        except Exception as e:
            print(f"[build_autoencoder VAE] Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            raise

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

    def save_full_vae(self, file_path):
        if self.autoencoder_model:
            # Need to provide custom_objects if Sampling layer is not automatically recognized
            self.autoencoder_model.save(file_path, custom_objects={'Sampling': Sampling})
            print(f"[save_full_vae] Full VAE model saved to {file_path}")
        else:
            print("Full VAE model not available to save.")

    def load_full_vae(self, file_path):
        self.autoencoder_model = load_model(file_path, custom_objects={'Sampling': Sampling})
        print(f"[load_full_vae] Full VAE model loaded from {file_path}")
        # After loading the full VAE, you might need to re-extract references to encoder and decoder parts
        # if your plugins expect self.encoder_model and self.decoder_model to be populated.
        # This can be done by finding layers by name.
        try:
            self.encoder_model = self.autoencoder_model.get_layer('vae_encoder') # Name given during VAE encoder creation
            self.decoder_model = self.autoencoder_model.get_layer('vae_decoder') # Name given during VAE decoder creation
            # Also, re-populate plugin's internal models if they are used elsewhere
            self.encoder_plugin.encoder_model = self.encoder_model
            self.decoder_plugin.model = self.decoder_model
            # And re-populate shape_before_flatten for decoder plugin
            self.encoder_plugin.load(None) # Call a modified load or a new method to re-init internal params like shape_before_flatten
            encoder_shape_before_flatten = self.encoder_plugin.shape_before_flatten_for_decoder
            if encoder_shape_before_flatten:
                 self.decoder_plugin.params['initial_dense_target_shape'] = encoder_shape_before_flatten

            print("[load_full_vae] References to VAE encoder and decoder parts updated.")
        except ValueError as e:
            print(f"[load_full_vae] Warning: Could not re-link encoder/decoder parts from loaded VAE model. Error: {e}")

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





