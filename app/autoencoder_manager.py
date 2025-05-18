import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Lambda, Layer 
from keras.optimizers import Adam
import keras 

from app.autoencoder_helper import (
    reconstruction_and_stats_loss_fn, 
    get_metrics,
    EarlyStoppingWithPatienceCounter,
    ReduceLROnPlateauWithCounter,
    KLAnnealingCallback # Import the new callback
)

class KLDivergenceLayer(Layer):
    def __init__(self, kl_beta=1.0, name="kl_divergence_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        # Make kl_beta a tf.Variable so it can be updated by the callback
        self.kl_beta = tf.Variable(initial_value=float(kl_beta), trainable=False, dtype=tf.float32, name="kl_beta_internal")

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss_val = -0.5 * keras.ops.sum(1 + z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var), axis=-1)
        mean_kl_loss_val = keras.ops.mean(kl_loss_val)
        
        weighted_kl_loss = self.kl_beta * mean_kl_loss_val # Use the tf.Variable
        self.add_loss(weighted_kl_loss)
        # Re-add metric for monitoring if desired
        self.add_metric(weighted_kl_loss, name="kl_divergence_value") 
        
        return z_mean # Layer must return a tensor

    def compute_output_shape(self, input_shape):
        # This layer returns z_mean, so its output shape is the shape of z_mean.
        # input_shape is a list of two shapes: [z_mean_shape, z_log_var_shape]
        if isinstance(input_shape, list) and len(input_shape) > 0:
            return input_shape[0]
        # Fallback or raise error if input_shape is not as expected
        # For Keras 3, often the shape is inferred, but explicit is safer.
        # If z_mean is the first input to this layer (if it were a single tensor input)
        # return input_shape 
        # However, since inputs is a list [z_mean, z_log_var], input_shape[0] is correct.
        raise ValueError(f"Unexpected input_shape format for KLDivergenceLayer: {input_shape}")


    def get_config(self):
        config = super().get_config()
        # Save the current value of kl_beta as a float
        config.update({"kl_beta": float(self.kl_beta.numpy())}) 
        return config

class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin):
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.autoencoder_model = None
        self.encoder_model = None # Component model
        self.decoder_model = None # Component model
        self.model = None # Alias for autoencoder_model
        self.kl_layer_instance = None # Add attribute to store KL layer
        print(f"[AutoencoderManager] Initialized for CVAE components.")

    def build_autoencoder(self, config):
        try:
            print("[build_autoencoder] Starting to build single-step CVAE...")
            window_size = config.get('window_size')
            input_features_per_step = config.get('input_features_per_step')
            rnn_hidden_dim = config.get('rnn_hidden_dim')
            conditioning_dim = config.get('conditioning_dim')
            latent_dim = config.get('latent_dim')
            output_feature_dim = 6 # Fixed for this CVAE

            if not all(v is not None for v in [window_size, input_features_per_step, rnn_hidden_dim, conditioning_dim, latent_dim]):
                raise ValueError("Config must provide 'window_size', 'input_features_per_step', 'rnn_hidden_dim', 'conditioning_dim', and 'latent_dim'.")
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
            input_x_window = Input(shape=(window_size, input_features_per_step), name="cvae_input_x_window")
            input_h_context = Input(shape=(rnn_hidden_dim,), name="cvae_input_h_context")
            input_conditions_t = Input(shape=(conditioning_dim,), name="cvae_input_conditions_t")

            # 4. Pass inputs through Encoder
            z_mean_raw, z_log_var_raw = self.encoder_model([input_x_window, input_h_context, input_conditions_t])

            # Explicitly name the output tensors from the encoder using Lambda layers
            z_mean = Lambda(lambda x: x, name='z_mean_output')(z_mean_raw)
            z_log_var = Lambda(lambda x: x, name='z_log_var_output')(z_log_var_raw)

            # 5. Sampling Layer
            def sampling_fn(args): 
                z_mean_sample, z_log_var_sample = args
                batch = tf.shape(z_mean_sample)[0]
                dim   = tf.shape(z_mean_sample)[1]
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean_sample + tf.exp(0.5 * z_log_var_sample) * epsilon
            
            z_sampled = Lambda(sampling_fn, output_shape=(latent_dim,), name='cvae_sampling_z')([z_mean, z_log_var])

            # 6. Pass z and context to Decoder
            reconstruction_raw = self.decoder_model([z_sampled, input_h_context, input_conditions_t])
            
            reconstruction = Lambda(lambda x: x, name='reconstruction_output')(reconstruction_raw)

            # Use kl_beta_start for initial KLDivergenceLayer instantiation
            # The KLAnnealingCallback will update this layer's kl_beta attribute
            initial_kl_beta = config.get('kl_beta_start', 0.0001) 
            # Store the instance of KLDivergenceLayer
            self.kl_layer_instance = KLDivergenceLayer(kl_beta=initial_kl_beta, name="kl_loss_adder_node")
            _ = self.kl_layer_instance([z_mean, z_log_var]) # Call the layer

            # 7. Create the CVAE Model
            self.autoencoder_model = Model(
                inputs=[input_x_window, input_h_context, input_conditions_t],
                outputs={
                    'reconstruction_output': reconstruction, 
                    'z_mean_output': z_mean,       
                    'z_log_var_output': z_log_var  
                },
                name="windowed_input_cvae_6_features_out"
            )
            self.model = self.autoencoder_model
            
            print("[build_autoencoder] Single-step CVAE model assembled.")
            self.autoencoder_model.summary(line_length=150)

            # === DEBUG PRINTS START ===
            print("\n" + "="*30 + " DEBUG INFO " + "="*30)
            print(f"DEBUG: autoencoder_model object: {self.autoencoder_model}")
            if self.autoencoder_model is not None:
                print(f"DEBUG: autoencoder_model.name: {self.autoencoder_model.name}")
                print(f"DEBUG: autoencoder_model.inputs: {self.autoencoder_model.inputs}")
                print(f"DEBUG: autoencoder_model.outputs: {self.autoencoder_model.outputs}") 
                try:
                    print(f"DEBUG: autoencoder_model.output_names: {self.autoencoder_model.output_names}") 
                except Exception as e:
                    print(f"DEBUG: Error accessing autoencoder_model.output_names: {e}")
                
                print(f"DEBUG: Tensors used in Model's outputs dict construction:")
                # Corrected debug prints:
                print(f"DEBUG:   'reconstruction_output' tensor: {reconstruction} (Tensor name: {reconstruction.name})")
                print(f"DEBUG:   'z_mean_output' tensor: {z_mean} (Tensor name: {z_mean.name})")
                print(f"DEBUG:   'z_log_var_output' tensor: {z_log_var} (Tensor name: {z_log_var.name})")

            print("="*72 + "\n")
            # === DEBUG PRINTS END ===

            adam_optimizer = Adam(
                learning_rate=config.get('learning_rate', 0.0001),
                beta_1=config.get('beta_1', 0.9),
                beta_2=config.get('beta_2', 0.999),
                epsilon=config.get('epsilon', 1e-7),
                amsgrad=config.get('amsgrad', False)
            )

            reconstruction_metrics = get_metrics(config=config) 
            print(f"DEBUG: Metrics for 'reconstruction_output': {reconstruction_metrics}")
            
            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss={
                    'reconstruction_output': reconstruction_and_stats_loss_fn,
                },
                metrics={ 
                    'reconstruction_output': reconstruction_metrics
                },
                run_eagerly=config.get('run_eagerly', False)
            )
            print("[build_autoencoder] Single-step CVAE model compiled successfully.")

        except Exception as e:
            print(f"[build_autoencoder] Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'autoencoder_model') and self.autoencoder_model is not None:
                print("\n" + "="*30 + " DEBUG INFO ON EXCEPTION " + "="*30)
                print(f"DEBUG (exception): autoencoder_model.name: {self.autoencoder_model.name}")
                print(f"DEBUG (exception): autoencoder_model.outputs: {self.autoencoder_model.outputs}")
                try:
                    print(f"DEBUG (exception): autoencoder_model.output_names: {self.autoencoder_model.output_names}")
                except Exception as e_inner:
                    print(f"DEBUG (exception): Error accessing autoencoder_model.output_names: {e_inner}")
                print("="*86 + "\n")
            raise
    def train_autoencoder(self, data_inputs, data_targets, epochs=100, batch_size=128, config=None):
        if config is None: config = {}
        
        if not self.autoencoder_model:
            raise RuntimeError("[train_autoencoder] CVAE model not built. Please call build_autoencoder first.")
        
        if not self.kl_layer_instance:
            print("[train_autoencoder] Warning: KLDivergenceLayer instance not found in AutoencoderManager. KLAnnealing might not work.")

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
        patience_es = config.get('early_patience', 30) # Default from your config
        # start_epoch_es = config.get('start_from_epoch', 10) # Not directly used by ES callback
        patience_rlr = config.get("reduce_lr_patience", max(1, int(patience_es / 3))) # Adjusted

        # KL Annealing parameters from config
        kl_beta_start = float(config.get('kl_beta_start', 0.0001)) 
        kl_beta_end = float(config.get('kl_beta', 1.0)) 
        kl_anneal_epochs = int(config.get('kl_anneal_epochs', 100)) 

        callbacks_list = [
            EarlyStoppingWithPatienceCounter(
               monitor='val_loss', patience=patience_es, restore_best_weights=True,
               verbose=1, min_delta=min_delta_es
            ),
            ReduceLROnPlateauWithCounter(
               monitor='val_loss', factor=0.5, patience=patience_rlr, cooldown=max(1, int(patience_rlr / 2)),
               min_delta=min_delta_es, verbose=1
            ),
            KLAnnealingCallback( # Add the new callback
                kl_beta_start=kl_beta_start,
                kl_beta_end=kl_beta_end,
                anneal_epochs=kl_anneal_epochs,
                kl_layer_instance=self.kl_layer_instance, # Pass the layer instance
                layer_name="kl_loss_adder_node", # Keep as fallback or for reference
                verbose=1 
            )
        ]
        
        # Determine validation data source
        validation_data = None
        validation_split_ratio = config.get('validation_split', 0.0) # Default to 0 if not specified

        if 'cvae_val_inputs' in config and 'cvae_val_targets' in config:
            # User has provided pre-split validation data via config (e.g., from data_processor)
            val_inputs_list = [
                config['cvae_val_inputs']['x_window'],
                config['cvae_val_inputs']['h_context'],
                config['cvae_val_inputs']['conditions_t']
            ]
            val_targets_dict = {'reconstruction_output': config['cvae_val_targets']}
            validation_data = (val_inputs_list, val_targets_dict)
            print(f"[train_autoencoder] Using pre-defined validation data. Shapes: "
                  f"x_window: {val_inputs_list[0].shape}, h_context: {val_inputs_list[1].shape}, "
                  f"conditions_t: {val_inputs_list[2].shape}, targets: {val_targets_dict['reconstruction_output'].shape}")
            validation_split_ratio = 0.0 # Do not use validation_split if validation_data is provided
        elif validation_split_ratio > 0:
            print(f"[train_autoencoder] Using validation_split: {validation_split_ratio}")
        else:
            print("[train_autoencoder] No validation_split and no pre-defined validation_data. Training without validation monitoring during fit.")


        history = self.autoencoder_model.fit(
            x=data_inputs,
            y={'reconstruction_output': data_targets}, 
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks_list,
            validation_split=validation_split_ratio if validation_data is None else None, # Only if not using validation_data
            validation_data=validation_data # Pass pre-split validation data if available
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
        # Ensure data_inputs is a list of 3 arrays for the model's inputs
        if not isinstance(data_inputs, list) or len(data_inputs) != 3:
             raise ValueError(f"[evaluate] data_inputs for {dataset_name} must be a list of 3 arrays.")

        results = self.autoencoder_model.evaluate(
            x=data_inputs,
            y={'reconstruction_output': data_targets}, # Pass y as a dictionary
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
        loaded_keras_model = load_model(file_path, compile=False, custom_objects={'KLDivergenceLayer': KLDivergenceLayer}) # if KLD is part of encoder
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
        loaded_keras_model = load_model(file_path, compile=False, custom_objects={'KLDivergenceLayer': KLDivergenceLayer}) # if KLD is part of decoder
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





