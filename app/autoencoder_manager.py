import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Lambda
from keras.optimizers import Adam

# pull in all loss/metric/callback helpers
from app.autoencoder_helper import (
    combined_cvae_loss_fn,
    get_metrics,
    EarlyStoppingWithPatienceCounter,
    ReduceLROnPlateauWithCounter
)


class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin):
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.autoencoder_model = None # This will be the single-step CVAE model
        self.encoder_model = None # Component model from encoder_plugin
        self.decoder_model = None # Component model from decoder_plugin
        self.model = None # Points to self.autoencoder_model
        print(f"[AutoencoderManager] Initialized for Sequential CVAE components.")

    def build_autoencoder(self, config):
        try:
            print("[build_autoencoder] Starting to build single-step CVAE with Conv1D Encoder and 6-feature Decoder...")

            # Get dimensions from config
            # For Encoder Input (Windowed Data)
            window_size = config.get('window_size')
            input_features_per_step = config.get('input_features_per_step') # Full features in the input window

            # For CVAE context and latent space
            rnn_hidden_dim = config.get('rnn_hidden_dim') # For h_context
            conditioning_dim = config.get('conditioning_dim') # For general conditions_t (e.g., previous 6 targets)
            latent_dim = config.get('latent_dim')

            # For Decoder Output (and CVAE target)
            output_feature_dim = 6 # Hardcoded as per requirement for OHLC + 2 derived

            if not all(v is not None for v in [window_size, input_features_per_step, rnn_hidden_dim, conditioning_dim, latent_dim]):
                raise ValueError(
                    "Config must provide 'window_size', 'input_features_per_step', 'rnn_hidden_dim', "
                    "'conditioning_dim', and 'latent_dim'."
                )
            if not isinstance(window_size, int) or window_size <= 0:
                raise ValueError(f"'window_size' must be a positive integer. Got: {window_size}")
            if not isinstance(input_features_per_step, int) or input_features_per_step <= 0:
                raise ValueError(f"'input_features_per_step' must be a positive integer. Got: {input_features_per_step}")


            # 1. Configure and get Encoder (Processes windowed input)
            self.encoder_plugin.configure_model_architecture(
                window_size=window_size,
                input_features_per_step=input_features_per_step,
                rnn_hidden_dim=rnn_hidden_dim,
                conditioning_dim=conditioning_dim,
                latent_dim=latent_dim,
                config=config
            )
            self.encoder_model = getattr(self.encoder_plugin, 'inference_network_model', None)
            if self.encoder_model is None:
                raise ValueError("Encoder plugin failed to build its model (inference_network_model).")
            print("[build_autoencoder] Encoder component (Conv1D-based) built.")
            self.encoder_model.summary(line_length=120)

            # 2. Configure and get Decoder (Generates 6 features)
            self.decoder_plugin.configure_model_architecture(
                latent_dim=latent_dim,
                rnn_hidden_dim=rnn_hidden_dim,
                conditioning_dim=conditioning_dim,
                output_feature_dim=output_feature_dim, # Should be 6, plugin enforces this
                config=config
            )
            self.decoder_model = getattr(self.decoder_plugin, 'generative_network_model', None)
            if self.decoder_model is None:
                raise ValueError("Decoder plugin failed to build its model (generative_network_model).")
            print(f"[build_autoencoder] Decoder component (outputting {output_feature_dim} features) built.")
            self.decoder_model.summary(line_length=120)

            # 3. Define Inputs for the combined Single-Step CVAE model
            # Input for the encoder part of CVAE
            input_x_window = Input(shape=(window_size, input_features_per_step), name="cvae_input_x_window")
            # Context inputs, shared by encoder and decoder logic
            input_h_context = Input(shape=(rnn_hidden_dim,), name="cvae_input_h_context")
            input_conditions_t = Input(shape=(conditioning_dim,), name="cvae_input_conditions_t")

            # 4. Pass inputs through Encoder
            # Encoder plugin expects [x_window, h_context, conditions_t]
            z_mean, z_log_var = self.encoder_model([input_x_window, input_h_context, input_conditions_t])

            # 5. Sampling Layer
            def sampling(args):
                z_mean_sample, z_log_var_sample = args
                batch = tf.shape(z_mean_sample)[0]
                dim   = tf.shape(z_mean_sample)[1]
                # draw epsilon from N(0,1)
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean_sample + tf.exp(0.5 * z_log_var_sample) * epsilon
            
            z = Lambda(sampling, output_shape=(latent_dim,), name='cvae_sampling_z')([z_mean, z_log_var])

            # 6. Pass z and context to Decoder
            # Decoder plugin expects [z, h_context, conditions_t]
            reconstruction = self.decoder_model([z, input_h_context, input_conditions_t]) # Output shape (batch, 6)

            # 7. Create the Single-Step CVAE Model
            # Define model with named outputs
            self.autoencoder_model = Model(
                inputs=[input_x_window, input_h_context, input_conditions_t], # CVAE takes the window as x_input
                outputs={
                    'reconstruction_output': reconstruction, # This will be y_pred in the reconstruction_loss_fn
                    'z_mean_output': z_mean,                 # For KL metric
                    'z_log_var_output': z_log_var            # For KL metric
                },
                name="windowed_input_cvae_6_features_out"
            )
            self.model = self.autoencoder_model 
            
            # KL divergence will be calculated within the main loss function
            # self.autoencoder_model.add_loss(weighted_kl_loss_for_model) # This line is removed

            print("[build_autoencoder] Single-step CVAE model assembled.")
            self.autoencoder_model.summary(line_length=150)

            # 8. Compile the CVAE Model
            adam_optimizer = Adam(
                learning_rate=config.get('learning_rate', 0.0001),
                beta_1=config.get('beta_1', 0.9),
                beta_2=config.get('beta_2', 0.999),
                epsilon=config.get('epsilon', 1e-7),
                amsgrad=config.get('amsgrad', False)
            )

            # compile with a single CVAE loss and standardized metrics
            self.autoencoder_model.compile(
                optimizer=Adam(config.get('learning_rate', 1e-4)),
                loss=combined_cvae_loss_fn,
                metrics=get_metrics(),
                run_eagerly=config.get('run_eagerly', False)
            )
            print("[build_autoencoder] Single-step CVAE model compiled successfully.")

        except Exception as e:
            print(f"[build_autoencoder] Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train_autoencoder(self, data_inputs, data_targets, epochs=100, batch_size=128, config=None):
        if config is None: config = {}
        
        if not self.autoencoder_model:
            raise RuntimeError("[train_autoencoder] Single-step CVAE model not built. Please call build_autoencoder first.")

        if not isinstance(data_inputs, (list, tuple)) or len(data_inputs) != 3:
            raise ValueError("data_inputs must be a list/tuple of 3 arrays: [x_window_data, h_context_data, conditions_t_data]")
        
        print(f"[train_autoencoder] Starting CVAE training.")
        print(f"Input data shapes: x_window: {data_inputs[0].shape}, h_context: {data_inputs[1].shape}, conditions_t: {data_inputs[2].shape}")
        print(f"Target data shape (6 features): {data_targets.shape}")

        if data_targets.shape[-1] != 6:
            raise ValueError(f"data_targets should have 6 features, but got shape {data_targets.shape}")

        if np.isnan(data_inputs[0]).any() or np.isnan(data_inputs[1]).any() or np.isnan(data_inputs[2]).any() or np.isnan(data_targets).any():
            raise ValueError("[train_autoencoder] Training data or targets contain NaN values.")
            
        min_delta_es = config.get("min_delta", 1e-7)
        patience_es = config.get('early_patience', 10)
        start_epoch_es = config.get('start_from_epoch', 10)
        patience_rlr = config.get("reduce_lr_patience", max(1, int(patience_es / 4)))

        callbacks_list = [
            EarlyStoppingWithPatienceCounter(
               monitor='val_loss',
               patience=patience_es,
               restore_best_weights=True,
               verbose=1,
               min_delta=min_delta_es
            ),
            ReduceLROnPlateauWithCounter(
               monitor='val_loss',
               factor=0.5,
               patience=patience_rlr,
               cooldown=5,
               min_delta=min_delta_es,
               verbose=1
            )
        ]

        # When using a single loss function for a model with multiple outputs,
        # y should be a dictionary mapping output names to targets if Keras needs to
        # align them. However, if the loss function itself handles the dict of y_preds,
        # y_true can be the direct target for the primary output (reconstruction).
        # Keras will pass y_true as is, and y_pred as a dict of model outputs.
        history = self.autoencoder_model.fit(
            x=data_inputs,
            y=data_targets,      # raw (batch,6) targets
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks_list,
            validation_split=config.get('validation_split', 0.2)
        )
        print(f"[train_autoencoder] CVAE Training loss: {history.history['loss'][-1] if history.history['loss'] else 'N/A'}")
        print("[train_autoencoder] CVAE Training completed.")
        return history

    def encode_data(self, per_step_encoder_inputs, config=None):
        if not self.encoder_model:
            if self.autoencoder_model and self.encoder_plugin and hasattr(self.encoder_plugin, 'inference_network_model'):
                 self.encoder_model = self.encoder_plugin.inference_network_model
            else:
                raise ValueError("[encode_data] Encoder component model not available.")
        
        # Encoder plugin expects a list: [x_window_batch, h_prev_batch, conditions_t_batch]
        if not isinstance(per_step_encoder_inputs, list) or len(per_step_encoder_inputs) != 3:
            raise ValueError("[encode_data] 'per_step_encoder_inputs' must be a list of three arrays: [x_window_batch, h_context_batch, conditions_t_batch].")
        
        print(f"[encode_data] Encoding with Conv1D encoder. Input part shapes: {[d.shape for d in per_step_encoder_inputs]}")
        z_mean, z_log_var = self.encoder_plugin.encode(per_step_encoder_inputs) 
        print(f"[encode_data] Encoder produced z_mean shape: {z_mean.shape}, z_log_var shape: {z_log_var.shape}")
        return z_mean, z_log_var

    def decode_data(self, per_step_decoder_inputs, config=None):
        if not self.decoder_model:
            if self.autoencoder_model and self.decoder_plugin and hasattr(self.decoder_plugin, 'generative_network_model'):
                self.decoder_model = self.decoder_plugin.generative_network_model
            else:
                raise ValueError("[decode_data] Decoder component model not available.")

        # Decoder plugin expects a list: [z_t_batch, h_t_batch, conditions_t_batch]
        if not isinstance(per_step_decoder_inputs, list) or len(per_step_decoder_inputs) != 3:
            raise ValueError("[decode_data] 'per_step_decoder_inputs' must be a list of three arrays: [z_t_batch, h_context_batch, conditions_t_batch].")

        print(f"[decode_data] Decoding with 6-feature decoder. Input part shapes: {[d.shape for d in per_step_decoder_inputs]}")
        reconstructed_x_t = self.decoder_plugin.decode(per_step_decoder_inputs) # Output shape (batch, 6)
        print(f"[decode_data] Decoder produced reconstructed x_t shape: {reconstructed_x_t.shape}")
        return reconstructed_x_t

    def evaluate(self, data_inputs, data_targets, dataset_name, config=None):
        if config is None: config = {}
        if not self.autoencoder_model:
            print(f"[evaluate] CVAE model evaluation skipped for {dataset_name}: Model not available.")
            return None 
        
        print(f"[evaluate] Evaluating CVAE on {dataset_name}.")
        results = self.autoencoder_model.evaluate(
            x=data_inputs,
            y=data_targets,         # raw reconstruction targets
            verbose=1,
            batch_size=config.get('batch_size', 128)
        )
        # results is a list: [total_loss, mae_on_reconstruction, huber_metric, kl_metric, mmd_metric, ...]
        # MAE is typically the first metric after loss if 'mae' was in the metrics list at compile time
        # and Keras applies it to the first output.
        # The order in results matches the order in metrics_list during compile, plus the loss.
        metrics_names = self.autoencoder_model.metrics_names
        results_dict = dict(zip(metrics_names, results))
        
        print(f"[evaluate] CVAE {dataset_name} Evaluation Results: {results_dict}")
        return results_dict # Return the dictionary of metrics

    def save_encoder(self, file_path):
        if self.encoder_model: # Save the component model
            self.encoder_model.save(file_path)
            print(f"[save_encoder] Encoder component model saved to {file_path}")
        elif self.encoder_plugin and hasattr(self.encoder_plugin, 'inference_network_model') and self.encoder_plugin.inference_network_model:
            self.encoder_plugin.inference_network_model.save(file_path) # Save from plugin if manager's copy is not set
            print(f"[save_encoder] Encoder component model (from plugin) saved to {file_path}")
        else:
            print("[save_encoder] Encoder component model not available to save.")

    def save_decoder(self, file_path):
        if self.decoder_model: # Save the component model
            self.decoder_model.save(file_path)
            print(f"[save_decoder] Decoder component model saved to {file_path}")
        elif self.decoder_plugin and hasattr(self.decoder_plugin, 'generative_network_model') and self.decoder_plugin.generative_network_model:
            self.decoder_plugin.generative_network_model.save(file_path) # Save from plugin
            print(f"[save_decoder] Decoder component model (from plugin) saved to {file_path}")
        else:
            print("[save_decoder] Decoder component model not available to save.")

    def load_encoder(self, file_path):
        # This method should load the Keras model and also inform the plugin
        loaded_keras_model = load_model(file_path, compile=False)
        self.encoder_model = loaded_keras_model # Manager holds a direct reference
        print(f"[load_encoder] Encoder component Keras model loaded from {file_path}")

        if hasattr(self.encoder_plugin, 'load') and callable(getattr(self.encoder_plugin, 'load')):
            try:
                # Plugin's load method should set its internal model and params
                self.encoder_plugin.load(file_path, compile_model=False)
                # Ensure manager's component model is also updated if plugin re-instantiates
                if hasattr(self.encoder_plugin, 'inference_network_model'):
                    self.encoder_model = self.encoder_plugin.inference_network_model
            except Exception as e:
                print(f"[load_encoder] Error calling encoder_plugin's load method: {e}. Keras model loaded into manager.")
        elif hasattr(self.encoder_plugin, 'inference_network_model'): # Fallback
            self.encoder_plugin.inference_network_model = loaded_keras_model
            print("[load_encoder] Keras model assigned to encoder_plugin.inference_network_model by manager.")


    def load_decoder(self, file_path):
        loaded_keras_model = load_model(file_path, compile=False)
        self.decoder_model = loaded_keras_model
        print(f"[load_decoder] Decoder component Keras model loaded from {file_path}")

        if hasattr(self.decoder_plugin, 'load') and callable(getattr(self.decoder_plugin, 'load')):
            try:
                self.decoder_plugin.load(file_path, compile_model=False)
                if hasattr(self.decoder_plugin, 'generative_network_model'):
                    self.decoder_model = self.decoder_plugin.generative_network_model
            except Exception as e:
                print(f"[load_decoder] Error calling decoder_plugin's load method: {e}. Keras model loaded into manager.")
        elif hasattr(self.decoder_plugin, 'generative_network_model'): # Fallback
            self.decoder_plugin.generative_network_model = loaded_keras_model
            print("[load_decoder] Keras model assigned to decoder_plugin.generative_network_model by manager.")

    def calculate_mse(self, original_data, reconstructed_data, config=None):
        print(f"[calculate_mse] Original data shape: {original_data.shape}")
        print(f"[calculate_mse] Reconstructed data shape: {reconstructed_data.shape}")
        if original_data.shape != reconstructed_data.shape:
            try:
                if np.prod(original_data.shape) == np.prod(reconstructed_data.shape):
                    reconstructed_data = reconstructed_data.reshape(original_data.shape)
                else: raise ValueError("Shape and element count mismatch.")
            except Exception as e:
                 raise ValueError(f"Shape mismatch: original {original_data.shape} vs reconstructed {reconstructed_data.shape}. Error: {e}")
        mse = np.mean(np.square(original_data - reconstructed_data))
        print(f"[calculate_mse] Calculated MSE: {mse}")
        return mse

    def calculate_mae(self, original_data, reconstructed_data, config=None):
        print(f"[calculate_mae] Original data shape: {original_data.shape}")
        print(f"[calculate_mae] Reconstructed data shape: {reconstructed_data.shape}")
        if original_data.shape != reconstructed_data.shape:
            try:
                if np.prod(original_data.shape) == np.prod(reconstructed_data.shape):
                    reconstructed_data = reconstructed_data.reshape(original_data.shape)
                else: raise ValueError("Shape and element count mismatch.")
            except Exception as e:
                raise ValueError(f"Shape mismatch: original {original_data.shape} vs reconstructed {reconstructed_data.shape}. Error: {e}")
        mae = tf.reduce_mean(tf.abs(tf.cast(original_data, tf.float32) - tf.cast(reconstructed_data, tf.float32))).numpy()
        print(f"[calculate_mae] Calculated MAE: {mae}")
        return mae





