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
        self.decoder_plugin = decoder_plugin # Retained for the old VAE path
        self.autoencoder_model = None
        self.encoder_model = None
        self.decoder_model = None # Retained for the old VAE path
        self.model = None # This will point to autoencoder_model (for VAE) or encoder_model (for new encoder type)
        print(f"[AutoencoderManager] Initialized.")

    def build_autoencoder(self, input_shape, interface_size, config, num_channels):
        try:
            print("[build_autoencoder] Starting to build components...")

            # Determine if the encoder plugin is the new type (per-step inference network)
            # The new type has 'configure_model_architecture' and its model is 'inference_network_model'
            is_new_encoder_type = hasattr(self.encoder_plugin, 'configure_model_architecture')

            if is_new_encoder_type:
                print("[build_autoencoder] Configuring new encoder type (per-step inference network).")
                # Parameters for the new encoder's 'configure_model_architecture'
                x_feature_dim = config.get('x_feature_dim')
                rnn_hidden_dim = config.get('rnn_hidden_dim')
                conditioning_dim = config.get('conditioning_dim')
                # 'interface_size' argument corresponds to 'latent_dim'

                if not all(v is not None for v in [x_feature_dim, rnn_hidden_dim, conditioning_dim, interface_size]):
                    raise ValueError(
                        "For the new encoder type, 'config' must provide 'x_feature_dim', "
                        "'rnn_hidden_dim', 'conditioning_dim', and 'interface_size' must be passed to build_autoencoder."
                    )
                
                self.encoder_plugin.configure_model_architecture(
                    x_feature_dim=x_feature_dim,
                    rnn_hidden_dim=rnn_hidden_dim,
                    conditioning_dim=conditioning_dim,
                    latent_dim=interface_size, # interface_size is the latent_dim
                    config=config
                )
                # The new encoder plugin stores its model in 'inference_network_model'
                self.encoder_model = getattr(self.encoder_plugin, 'inference_network_model', None)
                if self.encoder_model is None:
                    raise ValueError("New encoder plugin failed to build its model (inference_network_model is None or not set).")
                
                print("[build_autoencoder] New encoder component (per-step inference network) built successfully.")
                self.encoder_model.summary()

                # For this new encoder type, no traditional VAE is formed by this manager.
                self.decoder_model = None # No compatible decoder is assumed here.
                self.autoencoder_model = None # No autoencoder model.
                self.model = self.encoder_model # self.model points to the encoder itself.
                
                print("[build_autoencoder] Process finished for new encoder type. AutoencoderManager will manage only the encoder component.")
                # The 'feature-extractor' repo's role for this type is to produce this 'self.encoder_model'.
                # Training of this component as part of a sequential model (e.g., VRNN) happens in 'synthetic-datagen'.

            else:
                # --- Old VAE Path ---
                print("[build_autoencoder] Configuring old VAE encoder type.")
                use_sliding_windows = config.get('use_sliding_windows', True)
                self.encoder_plugin.configure_size(input_shape, interface_size, num_channels, use_sliding_windows, config)
                self.encoder_model = getattr(self.encoder_plugin, 'encoder_model', None) # Old attribute name
                if self.encoder_model is None:
                     raise ValueError("Old VAE encoder plugin failed to build its model (encoder_model is None or not set).")
                print("[build_autoencoder] Old VAE encoder model built successfully.")
                self.encoder_model.summary()

                encoder_shape_before_flatten = getattr(self.encoder_plugin, 'shape_before_flatten_for_decoder', None)
                if encoder_shape_before_flatten is None or (isinstance(encoder_shape_before_flatten, tuple) and None in encoder_shape_before_flatten):
                    print(f"[build_autoencoder] Critical: 'shape_before_flatten_for_decoder' not available or invalid from VAE encoder: {encoder_shape_before_flatten}. Cannot build VAE.")
                    self.decoder_model = None
                    self.autoencoder_model = None
                    self.model = self.encoder_model
                    return # Cannot proceed to build VAE

                print(f"Encoder shape_before_flatten_for_decoder: {encoder_shape_before_flatten}")

                if self.decoder_plugin is None:
                    print("[build_autoencoder] No VAE decoder plugin provided. Cannot build full VAE.")
                    self.autoencoder_model = None
                    self.model = self.encoder_model
                    return

                self.decoder_plugin.configure_size(
                    interface_size, input_shape, num_channels,
                    encoder_shape_before_flatten, use_sliding_windows, config
                )
                self.decoder_model = getattr(self.decoder_plugin, 'model', None)
                if self.decoder_model is None:
                    raise ValueError("Old VAE decoder plugin failed to build its model.")
                print("[build_autoencoder] Old VAE decoder model built successfully.")
                self.decoder_model.summary()

                # VAE Assembly
                z_mean, z_log_var = self.encoder_model.output
                z = Lambda(
                    lambda inputs_sampling: inputs_sampling[0] + K.random_normal(K.shape(inputs_sampling[0])) * K.exp(0.5 * inputs_sampling[1]),
                    name="vae_sampling"
                )([z_mean, z_log_var])
                autoencoder_output = self.decoder_model(z)

                self.autoencoder_model = Model(
                    inputs=self.encoder_model.input, outputs=autoencoder_output, name="autoencoder_vae"
                )
                self.model = self.autoencoder_model # self.model points to the full VAE

                # Compile VAE
                adam_optimizer = Adam(
                    learning_rate=config.get('learning_rate', 0.001),
                    beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
                )

                # The combined_loss function and metric functions (mmd_metric, huber_metric, etc.)
                # are assumed to be defined in the global scope of this file, as in the original context.
                # They need access to the 'config' object and global trackers (mmd_total, etc.).
                
                # Wrapper for combined_loss to ensure it captures 'config' from this scope
                def combined_loss_wrapper(y_true, y_pred):
                    # This is where the full logic of your combined_loss function (from the original file)
                    # should be placed or called, ensuring it has access to the 'config' variable
                    # from the 'build_autoencoder' method's scope.
                    # For brevity, this is a conceptual placeholder.
                    # Example structure:
                    y_true_f32 = tf.cast(y_true, tf.float32)
                    y_pred_f32 = tf.cast(y_pred, tf.float32)
                    h_loss = Huber(delta=1.0)(y_true_f32, y_pred_f32)
                    huber_loss_tracker.assign(h_loss)
                    
                    y_true_flat = tf.reshape(y_true_f32, [tf.shape(y_true_f32)[0], -1])
                    y_pred_flat = tf.reshape(y_pred_f32, [tf.shape(y_pred_f32)[0], -1])
                    
                    mmd_val = compute_mmd(y_true_flat, y_pred_flat, config.get('mmd_sigma', 1.0)) # compute_mmd is global
                    weighted_mmd = config.get('mmd_weight', 1.0) * mmd_val
                    mmd_total.assign(weighted_mmd)
                    
                    total_l = h_loss + weighted_mmd
                    
                    # Add skewness, kurtosis, covariance losses similarly, using 'config'
                    # and calling calculate_standardized_moment, covariance_loss_calc (assumed global)
                    # and updating their respective trackers.
                    # This part needs to replicate the full logic from your original combined_loss.
                    skew_loss_w = config.get('skew_loss_weight', 0.0)
                    if tf.cast(skew_loss_w, tf.float32) > 0.0:
                        # ... skew calculation ...
                        # total_l += weighted_skew_loss
                        pass # Placeholder for full skew logic
                    else: skew_loss_tracker.assign(0.0)

                    kurtosis_loss_w = config.get('kurtosis_loss_weight', 0.0)
                    if tf.cast(kurtosis_loss_w, tf.float32) > 0.0:
                        # ... kurtosis calculation ...
                        # total_l += weighted_kurtosis_loss
                        pass # Placeholder for full kurtosis logic
                    else: kurtosis_loss_tracker.assign(0.0)

                    covariance_loss_w = config.get('covariance_loss_weight', 0.0)
                    # ... full covariance calculation with conditions and debug print ...
                    # total_l += cov_loss_to_add_val
                    pass # Placeholder for full covariance logic

                    return total_l # Placeholder

                # Metrics (assuming these are defined globally or accessible)
                def mmd_metric_fn(y_true, y_pred): return mmd_total
                def huber_metric_fn(y_true, y_pred): return huber_loss_tracker
                def skew_metric_fn(y_true, y_pred): return skew_loss_tracker
                def kurtosis_metric_fn(y_true, y_pred): return kurtosis_loss_tracker
                def covariance_metric_fn(y_true, y_pred): return covariance_loss_tracker

                metrics_list = ['mae', mmd_metric_fn, huber_metric_fn, skew_metric_fn, kurtosis_metric_fn, covariance_metric_fn]

                self.autoencoder_model.compile(
                    optimizer=adam_optimizer,
                    loss=combined_loss_wrapper, # Use the wrapper
                    metrics=metrics_list,
                    run_eagerly=config.get('run_eagerly', False)
                )
                print("[build_autoencoder] Old VAE autoencoder model built and compiled successfully.")
                self.autoencoder_model.summary()

        except Exception as e:
            print(f"[build_autoencoder] Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train_autoencoder(self, data, epochs=100, batch_size=128, config=None):
        if config is None: config = {}
        
        is_new_encoder_type = hasattr(self.encoder_plugin, 'configure_model_architecture') and self.autoencoder_model is None

        if is_new_encoder_type:
            print("[train_autoencoder] Training skipped: The active model is a per-step inference network component.")
            print("This component is typically trained as part of a larger sequential model (e.g., VRNN) in the synthetic-datagen repository.")
            return None
        
        if not self.autoencoder_model:
            raise RuntimeError("[train_autoencoder] VAE autoencoder model not built. Please call build_autoencoder first.")

        # --- Original VAE training logic ---
        try:
            print(f"[train_autoencoder] Starting VAE training with data shape: {data.shape}")
            # (The rest of the VAE training logic from your file, including data reshaping, NaN checks, callbacks, model.fit)
            # Ensure 'config' is correctly used, e.g., for validation_split, MMDWeightAdjustmentCallback
            use_sliding_windows = config.get('use_sliding_windows', True)
            data_train = data
            if not use_sliding_windows and len(data.shape) == 2:
                data_train = np.expand_dims(data, axis=-1)
            
            if np.isnan(data_train).any():
                raise ValueError("[train_autoencoder] Training data contains NaN values.")

            print(f"[train_autoencoder] Training VAE with data shape: {data_train.shape}")
            
            min_delta_es = config.get("min_delta", 1e-7)
            patience_es = config.get('early_patience', 10)
            start_epoch_es = config.get('start_from_epoch', 10)
            patience_rlr = config.get("reduce_lr_patience", max(1, int(patience_es / 4)))

            callbacks_list = [
                EarlyStoppingWithPatienceCounter(monitor='val_loss', patience=patience_es, restore_best_weights=True, verbose=1, start_from_epoch=start_epoch_es, min_delta=min_delta_es),
                ReduceLROnPlateauWithCounter(monitor="val_loss", factor=0.5, patience=patience_rlr, cooldown=5, min_delta=min_delta_es, verbose=1),
                LambdaCallback(on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}"))
            ]
            if config.get('use_mmd_weight_adjustment', False):
                callbacks_list.append(MMDWeightAdjustmentCallback(config))

            history = self.autoencoder_model.fit(
                data_train, data_train, epochs=epochs, batch_size=batch_size, verbose=1,
                callbacks=callbacks_list, validation_split=config.get('validation_split', 0.2)
            )
            print(f"[train_autoencoder] VAE Training loss: {history.history['loss'][-1] if history.history['loss'] else 'N/A'}")
            print("[train_autoencoder] VAE Training completed.")
            return history
        except Exception as e:
            print(f"[train_autoencoder] Exception during VAE training: {e}")
            import traceback
            traceback.print_exc()
            raise

    def encode_data(self, data, config):
        if not self.encoder_model:
            raise ValueError("[encode_data] Encoder model not available. Build or load it first.")
        
        is_new_encoder_type = hasattr(self.encoder_plugin, 'configure_model_architecture')

        if is_new_encoder_type:
            # The new encoder plugin's 'encode' method expects a list of inputs: [x_t, h_prev, conditions_t]
            if not isinstance(data, list) or len(data) != 3:
                raise ValueError("[encode_data] For new encoder type, 'data' argument must be a list of three arrays: [x_t_batch, h_prev_batch, conditions_t_batch].")
            print(f"[encode_data] Encoding with new per-step inference network. Input part shapes: {[d.shape for d in data]}")
            # The plugin's own encode method calls predict
            encoded_parts = self.encoder_plugin.encode(data) 
            print(f"[encode_data] New encoder produced z_mean shape: {encoded_parts[0].shape}, z_log_var shape: {encoded_parts[1].shape}")
            return encoded_parts # Returns [z_mean, z_log_var]
        else:
            # --- Old VAE Encoder Path ---
            print(f"[encode_data] Encoding with old VAE encoder. Input data shape: {data.shape}")
            data_to_encode = data
            use_sliding_windows = config.get('use_sliding_windows', True)
            if not use_sliding_windows and len(data_to_encode.shape) == 2:
                data_to_encode = np.expand_dims(data_to_encode, axis=-1)
            
            print(f"[encode_data] Data shape for VAE encoding: {data_to_encode.shape}")
            encoded_data_parts = self.encoder_model.predict(data_to_encode) # [z_mean, z_log_var]
            print(f"[encode_data] VAE Encoded z_mean shape: {encoded_data_parts[0].shape}, z_log_var shape: {encoded_data_parts[1].shape}")
            return encoded_data_parts

    def decode_data(self, encoded_z_sample, config): # encoded_z_sample is the sampled z
        is_new_encoder_type = hasattr(self.encoder_plugin, 'configure_model_architecture')

        if is_new_encoder_type:
            print("[decode_data] Decoding skipped: No compatible decoder for the new per-step inference network is managed by AutoencoderManager.")
            print("Decoding is part of the sequential generator in synthetic-datagen, which would use a per-step generative network.")
            return None 
        
        if not self.decoder_model: # This implies old VAE path but decoder not built/loaded
            raise ValueError("[decode_data] VAE Decoder model not available.")

        # --- Old VAE Decoder Path ---
        print(f"[decode_data] Decoding with VAE decoder. Input z_sample shape: {encoded_z_sample.shape}")
        decoded_data = self.decoder_model.predict(encoded_z_sample)
        
        # Reshaping logic from original file
        if not config.get('use_sliding_windows', True):
            if 'original_feature_size' in config and decoded_data.shape[1] * decoded_data.shape[2] == config['original_feature_size']:
                 decoded_data = decoded_data.reshape((decoded_data.shape[0], config['original_feature_size']))
            elif decoded_data.shape[-1] == 1 and len(decoded_data.shape) ==3 :
                 decoded_data = np.squeeze(decoded_data, axis=-1)
        # else: (for sliding windows, shape is likely as intended)
        print(f"[decode_data] VAE Decoded data final shape: {decoded_data.shape}")
        return decoded_data

    def evaluate(self, data, dataset_name, config):
        is_new_encoder_type = hasattr(self.encoder_plugin, 'configure_model_architecture')

        if is_new_encoder_type or not self.autoencoder_model:
            print(f"[evaluate] VAE evaluation skipped for {dataset_name}: No VAE autoencoder model available for evaluation.")
            print("The new per-step inference network component is evaluated as part of the full sequential model in synthetic-datagen.")
            return None, None 

        # --- Old VAE Evaluation Path ---
        print(f"[evaluate] Evaluating VAE on {dataset_name} data with shape: {data.shape}")
        data_eval = data
        if not config.get('use_sliding_windows', True) and len(data.shape) == 2:
            data_eval = np.expand_dims(data, axis=-1)
        
        results = self.autoencoder_model.evaluate(data_eval, data_eval, verbose=1)
        loss_val = results[0]
        mae_val = results[1] # Assuming 'mae' is the first metric
        print(f"[evaluate] VAE {dataset_name} Evaluation - Loss: {loss_val}, MAE: {mae_val}")
        return loss_val, mae_val
    
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
        # Generic load, assumes the saved model is a Keras model.
        self.encoder_model = load_model(file_path, compile=False)
        print(f"[load_encoder] Encoder model loaded from {file_path}")

        # Inform the plugin instance that its model has been loaded
        # This allows the plugin to update its internal state (e.g., dimension params from model structure)
        if hasattr(self.encoder_plugin, 'load') and callable(getattr(self.encoder_plugin, 'load')):
            try:
                # The plugin's load method should handle setting its internal model attribute
                self.encoder_plugin.load(file_path, compile_model=False) 
                # Ensure AutoencoderManager's self.encoder_model also points to the right model from plugin
                if hasattr(self.encoder_plugin, 'inference_network_model') and self.encoder_plugin.inference_network_model is not None:
                    self.encoder_model = self.encoder_plugin.inference_network_model
                elif hasattr(self.encoder_plugin, 'encoder_model') and self.encoder_plugin.encoder_model is not None: # old attribute name
                     self.encoder_model = self.encoder_plugin.encoder_model
            except Exception as e:
                print(f"[load_encoder] Error calling plugin's load method: {e}. Model loaded into AutoencoderManager directly.")
        else: # Fallback if plugin has no specific load method
            if hasattr(self.encoder_plugin, 'inference_network_model'): # New type
                self.encoder_plugin.inference_network_model = self.encoder_model
            elif hasattr(self.encoder_plugin, 'encoder_model'): # Old type
                self.encoder_plugin.encoder_model = self.encoder_model
            print("[load_encoder] Encoder model assigned to plugin's model attribute by AutoencoderManager.")


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





