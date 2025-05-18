import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Concatenate, Layer, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import json

# Import the new wrapper for the loss function
from app.autoencoder_helper import (
    get_reconstruction_and_stats_loss_fn, 
    get_metrics,
    EarlyStoppingWithPatienceCounter,
    ReduceLROnPlateauWithCounter,
    KLAnnealingCallback 
)

class KLDivergenceLayer(Layer):
    def __init__(self, kl_beta_start=0.001, name="kl_loss_adder_node", **kwargs):
        super(KLDivergenceLayer, self).__init__(name=name, **kwargs)
        # Ensure kl_beta_start is a float before creating tf.Variable
        self.kl_beta_val = float(kl_beta_start) 
        self.kl_beta = tf.Variable(self.kl_beta_val, trainable=False, dtype=tf.float32, name="kl_beta_weight")
        tf.print(f"KLDivergenceLayer '{self.name}' initialized with kl_beta_start: {self.kl_beta_val}")

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        weighted_kl_loss = self.kl_beta * kl_loss
        self.add_loss(weighted_kl_loss)
        
        # Add metrics to track KL components
        self.add_metric(kl_loss, name="kl_divergence_raw") 
        self.add_metric(weighted_kl_loss, name="kl_divergence_weighted") 
        self.add_metric(self.kl_beta, name="kl_beta_current") # Track current beta
        
        # tf.print(f"KLDivergenceLayer '{self.name}': kl_beta={self.kl_beta.numpy()}, kl_loss={kl_loss.numpy()}, weighted_kl_loss={weighted_kl_loss.numpy()}")
        return z_mean # Layer must return a tensor

    def compute_output_shape(self, input_shape):
        # This layer returns z_mean, so its output shape is the shape of z_mean.
        # input_shape is a list of two shapes: [z_mean_shape, z_log_var_shape]
        if isinstance(input_shape, list) and len(input_shape) > 0 and isinstance(input_shape[0], tuple):
            return input_shape[0]
        # Fallback or raise error if input_shape is not as expected
        tf.print(f"Warning: KLDivergenceLayer.compute_output_shape received unexpected input_shape: {input_shape}. Attempting to infer.")
        # Attempt to infer from the first element if it's a list/tuple of shapes
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and \
           isinstance(input_shape[0], (list, tuple, tf.TensorShape)):
             return tf.TensorShape(input_shape[0])
        if isinstance(input_shape, tf.TensorShape): # If only one shape is passed (should not happen with list input)
            return input_shape
        raise ValueError(f"Unexpected input_shape format for KLDivergenceLayer: {input_shape}")


    def get_config(self):
        config = super().get_config()
        # Save the current value of kl_beta as a float
        config.update({"kl_beta_start": float(self.kl_beta.numpy())}) 
        return config

class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin):
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.autoencoder_model = None
        self.encoder_model = None # Component model
        self.decoder_model = None # Component model
        self.model = None # Alias for autoencoder_model
        self.kl_layer_instance = None # Store KL layer instance
        tf.print(f"[AutoencoderManager] Initialized for CVAE components.")

    def build_autoencoder(self, config):
        try:
            tf.print("[build_autoencoder] Starting to build single-step CVAE...")
            
            # Define Input layers based on config
            window_size = config['window_size']
            num_features_input = config['num_features_input']
            rnn_hidden_dim = config['rnn_hidden_dim'] # Assuming this is h_context dim
            conditioning_dim = config['conditioning_dim'] # For conditions_t
            latent_dim = config['latent_dim']
            num_features_output = config['num_features_output']

            cvae_input_x_window = Input(shape=(window_size, num_features_input), name='cvae_input_x_window')
            cvae_input_h_context = Input(shape=(rnn_hidden_dim,), name='cvae_input_h_context')
            cvae_input_conditions_t = Input(shape=(conditioning_dim,), name='cvae_input_conditions_t')

            # Encoder part
            # Ensure encoder_plugin.create_encoder returns all necessary states if decoder needs them
            encoder_outputs, state_h_encoder, state_c_encoder = self.encoder_plugin.create_encoder(
                cvae_input_x_window, cvae_input_h_context, cvae_input_conditions_t, config
            )
            
            # Latent space
            z_mean = Dense(latent_dim, name='z_mean_output_dense')(encoder_outputs)
            z_log_var = Dense(latent_dim, name='z_log_var_output_dense')(encoder_outputs)

            # KL Divergence Layer
            kl_beta_start_from_config = config.get('kl_beta_start', 0.0001) # Get initial beta from main config
            self.kl_layer_instance = KLDivergenceLayer(kl_beta_start=kl_beta_start_from_config, name="kl_loss_adder_node")([z_mean, z_log_var])
            # The KLDivergenceLayer returns z_mean. The KL loss is added internally by the layer.
            
            # Sampling (reparameterization trick)
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            z_sampled = z_mean + tf.exp(0.5 * z_log_var) * epsilon

            # Decoder part
            # Pass necessary states from encoder to decoder if your architecture requires it
            reconstruction_output = self.decoder_plugin.create_decoder(
                z_sampled, state_h_encoder, cvae_input_conditions_t, config 
            )

            self.autoencoder_model = Model(
                inputs=[cvae_input_x_window, cvae_input_h_context, cvae_input_conditions_t],
                outputs={
                    'reconstruction_output': reconstruction_output,
                    'z_mean_output': z_mean,       # For analysis/debugging
                    'z_log_var_output': z_log_var  # For analysis/debugging
                },
                name=f"windowed_input_cvae_{num_features_output}_features_out"
            )
            self.model = self.autoencoder_model # Alias

            # --- Optimizer ---
            adam_optimizer = Adam(
                learning_rate=config.get('learning_rate', 0.0001),
                beta_1=config.get('adam_beta_1', 0.9),
                beta_2=config.get('adam_beta_2', 0.999),
                epsilon=config.get('adam_epsilon', 1e-07)
            )
            
            # --- Loss Function (Using the wrapper from autoencoder_helper) ---
            # Pass the main 'config' dictionary to the wrapper
            configured_loss_fn = get_reconstruction_and_stats_loss_fn(config) 

            # --- Metrics ---
            # Pass the main 'config' to get_metrics if any metric depends on it
            reconstruction_metrics = get_metrics(config=config) 

            tf.print(f"DEBUG: autoencoder_model object: {self.autoencoder_model}")
            tf.print(f"DEBUG: autoencoder_model.name: {self.autoencoder_model.name}")
            tf.print(f"DEBUG: autoencoder_model.inputs: {self.autoencoder_model.inputs}")
            tf.print(f"DEBUG: autoencoder_model.outputs: {self.autoencoder_model.outputs}")
            tf.print(f"DEBUG: autoencoder_model.output_names: {self.autoencoder_model.output_names}")
            tf.print(f"DEBUG: Metrics for 'reconstruction_output': {reconstruction_metrics}")

            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss={
                    'reconstruction_output': configured_loss_fn, 
                    # z_mean and z_log_var are intermediate; KL loss is added by KLDivergenceLayer
                    'z_mean_output': None, 
                    'z_log_var_output': None
                },
                loss_weights={ # Ensure only reconstruction_output contributes to primary loss sum
                    'reconstruction_output': 1.0,
                    'z_mean_output': 0.0,
                    'z_log_var_output': 0.0
                },
                metrics={ 
                    'reconstruction_output': reconstruction_metrics
                    # Keras will automatically pick up metrics added in KLDivergenceLayer
                },
                run_eagerly=config.get('run_eagerly', False) 
            )
            tf.print("[build_autoencoder] Single-step CVAE model compiled successfully.")

        except Exception as e:
            tf.print(f"Error during CVAE model building: {e}")
            import traceback
            tf.print(traceback.format_exc())
            raise
    
    def train_autoencoder(self, data_inputs, data_targets, epochs, batch_size, config):
        tf.print("[train_autoencoder] Starting CVAE training.")
        
        # Prepare input data dictionary based on model's input names
        train_inputs_dict = {
            'cvae_input_x_window': data_inputs[0],
            'cvae_input_h_context': data_inputs[1],
            'cvae_input_conditions_t': data_inputs[2]
        }
        # Prepare target data dictionary based on model's output names
        # KL loss is handled by the layer, so only reconstruction_output needs a real target.
        train_targets_dict = {
            'reconstruction_output': data_targets,
            # Dummy targets for z_mean and z_log_var as they don't have a direct loss here
            # 'z_mean_output': np.zeros((data_targets.shape[0], config['latent_dim'])), 
            # 'z_log_var_output': np.zeros((data_targets.shape[0], config['latent_dim']))
        }

        tf.print(f"Input data shapes: x_window: {data_inputs[0].shape}, h_context: {data_inputs[1].shape}, conditions_t: {data_inputs[2].shape}")
        tf.print(f"Target data shape for reconstruction_output ({config['num_features_output']} features): {data_targets.shape}")

        # Callbacks
        callbacks_list = []
        if config.get('early_stopping_patience', 0) > 0:
            early_stopping = EarlyStoppingWithPatienceCounter(
                monitor=config.get('early_stopping_monitor', 'val_loss'), # Default to val_loss
                patience=config.get('early_stopping_patience'),
                verbose=1,
                restore_best_weights=config.get('early_stopping_restore_best_weights', True)
            )
            callbacks_list.append(early_stopping)

        if config.get('reduce_lr_patience', 0) > 0:
            reduce_lr = ReduceLROnPlateauWithCounter(
                monitor=config.get('reduce_lr_monitor', 'val_loss'), # Default to val_loss
                factor=config.get('reduce_lr_factor', 0.2),
                patience=config.get('reduce_lr_patience'),
                min_lr=config.get('reduce_lr_min_lr', 0.00001),
                verbose=1
            )
            callbacks_list.append(reduce_lr)
        
        # KL Annealing Callback
        if self.kl_layer_instance and config.get('kl_anneal_epochs', 0) > 0:
            kl_annealing = KLAnnealingCallback(
                kl_beta_start=config.get('kl_beta_start', 0.0001), # Initial beta
                kl_beta_end=config.get('kl_beta', 1.0), # Target beta (often called just 'kl_beta' in config)
                anneal_epochs=config.get('kl_anneal_epochs'),
                kl_layer_instance=self.kl_layer_instance, # Pass the actual layer instance
                verbose=1
            )
            callbacks_list.append(kl_annealing)
            tf.print("[train_autoencoder] KL Annealing callback configured.")
        else:
            tf.print("[train_autoencoder] KL Annealing not configured (kl_layer_instance missing or kl_anneal_epochs is 0).")


        # Validation Data
        validation_data_prepared = None
        if 'cvae_val_inputs' in config and 'cvae_val_targets' in config:
            val_inputs_dict = {
                'cvae_input_x_window': config['cvae_val_inputs']['x_window'],
                'cvae_input_h_context': config['cvae_val_inputs']['h_context'],
                'cvae_input_conditions_t': config['cvae_val_inputs']['conditions_t']
            }
            val_targets_dict = {
                'reconstruction_output': config['cvae_val_targets'],
                # 'z_mean_output': np.zeros((config['cvae_val_targets'].shape[0], config['latent_dim'])),
                # 'z_log_var_output': np.zeros((config['cvae_val_targets'].shape[0], config['latent_dim']))
            }
            validation_data_prepared = (val_inputs_dict, val_targets_dict)
            tf.print(f"[train_autoencoder] Using pre-defined validation data. Shapes: "
                  f"x_window: {config['cvae_val_inputs']['x_window'].shape}, "
                  f"h_context: {config['cvae_val_inputs']['h_context'].shape}, "
                  f"conditions_t: {config['cvae_val_inputs']['conditions_t'].shape}, "
                  f"targets: {config['cvae_val_targets'].shape}")
        else:
            tf.print("[train_autoencoder] No pre-defined validation_data. Training without validation monitoring during fit if no validation_split is provided to Keras fit.")


        history = self.autoencoder_model.fit(
            train_inputs_dict,
            train_targets_dict,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            validation_data=validation_data_prepared, # Pass the prepared validation data
            shuffle=True, # Shuffle training data each epoch
            verbose=config.get('keras_verbose_level', 1) # Control Keras verbosity (1=progress bar, 2=one line per epoch)
        )
        return history

    def save_model(self, model_path, encoder_path=None, decoder_path=None):
        if self.autoencoder_model:
            tf.print(f"Saving full CVAE model to {model_path}")
            self.autoencoder_model.save(model_path)
            
            # Optionally save encoder and decoder separately
            # This requires constructing them as standalone models if not already done
            # For now, focus on saving the main model.
            if encoder_path and self.encoder_model: # Assuming self.encoder_model is built
                 tf.print(f"Saving encoder model to {encoder_path}")
                 self.encoder_model.save(encoder_path)
            if decoder_path and self.decoder_model: # Assuming self.decoder_model is built
                 tf.print(f"Saving decoder model to {decoder_path}")
                 self.decoder_model.save(decoder_path)
        else:
            tf.print("No model to save.")

    def load_model(self, model_path, custom_objects=None):
        tf.print(f"Loading CVAE model from {model_path}")
        if custom_objects is None:
            custom_objects = {
                'KLDivergenceLayer': KLDivergenceLayer,
                # Add the actual loss function name if it's custom and not a standard Keras one
                # 'reconstruction_and_stats_loss_fn_inner': get_reconstruction_and_stats_loss_fn(self.config_at_load_time) # This needs careful handling of config
            }
        # For loading, Keras needs to know about custom layers.
        # The loss function itself (if it was a simple function) is usually serialized by reference
        # or by its config if it's a Keras Loss class.
        # If using the wrapper, ensure Keras can find the function or provide it if needed.
        # Often, for custom function losses, Keras serializes its name.
        # It's safer to re-compile with the loss function if issues arise.
        
        self.autoencoder_model = load_model(model_path, custom_objects=custom_objects, compile=False) # Load without compiling first
        self.model = self.autoencoder_model
        
        # It's often necessary to re-compile the model after loading if it has custom losses/metrics
        # or if you want to change the optimizer. You'll need a config dictionary available here.
        # For now, we assume the user will re-compile if needed after loading.
        tf.print(f"Model {self.autoencoder_model.name} loaded. You may need to re-compile it with optimizer and loss.")
        
        # Try to find the KL layer instance after loading
        try:
            self.kl_layer_instance = self.autoencoder_model.get_layer("kl_loss_adder_node")
            tf.print(f"KL Divergence layer '{self.kl_layer_instance.name}' found in loaded model.")
        except ValueError:
            self.kl_layer_instance = None
            tf.print("KL Divergence layer 'kl_loss_adder_node' not found in loaded model.")


    def evaluate(self, data_inputs, data_targets, dataset_name="Test", config=None):
        if not self.autoencoder_model:
            tf.print("Model not built or loaded. Cannot evaluate.")
            return None
        
        tf.print(f"Evaluating model on {dataset_name} data...")
        eval_inputs_dict = {
            'cvae_input_x_window': data_inputs[0],
            'cvae_input_h_context': data_inputs[1],
            'cvae_input_conditions_t': data_inputs[2]
        }
        eval_targets_dict = {
            'reconstruction_output': data_targets
        }
        
        results = self.autoencoder_model.evaluate(
            eval_inputs_dict,
            eval_targets_dict,
            batch_size=config.get('batch_size', 128) if config else 128,
            verbose=config.get('keras_verbose_level', 1) if config else 1
        )
        
        tf.print(f"Evaluation results for {dataset_name}:")
        if isinstance(results, list): # Results is a list: [total_loss, output1_loss, output1_metric1, output1_metric2, ...]
            i = 0
            tf.print(f"  Total Loss: {results[i]}")
            i += 1
            for output_name in self.autoencoder_model.output_names: # Order matters
                # Check if this output has a loss configured (not None in compile)
                if self.autoencoder_model.loss_functions[self.autoencoder_model.output_names.index(output_name)] is not None:
                    tf.print(f"  Loss for '{output_name}': {results[i]}")
                    i += 1
                # Metrics for this output
                metrics_for_output = self.autoencoder_model.metrics_names[i : i + len(self.autoencoder_model.metrics[self.autoencoder_model.output_names.index(output_name)])]
                for metric_name, metric_val in zip(metrics_for_output, results[i : i + len(metrics_for_output)]):
                    tf.print(f"  Metric '{metric_name}' for '{output_name}': {metric_val}")
                i += len(metrics_for_output)
        else: # Single loss value
            tf.print(f"  Loss: {results}")
            
        return results

    def predict(self, data_inputs, config=None):
        if not self.autoencoder_model:
            tf.print("Model not built or loaded. Cannot predict.")
            return None
            
        tf.print("Generating predictions...")
        pred_inputs_dict = {
            'cvae_input_x_window': data_inputs[0],
            'cvae_input_h_context': data_inputs[1],
            'cvae_input_conditions_t': data_inputs[2]
        }
        
        predictions = self.autoencoder_model.predict(
            pred_inputs_dict,
            batch_size=config.get('batch_size', 128) if config else 128,
            verbose=config.get('keras_verbose_level', 1) if config else 1
        )
        # Predictions will be a dictionary if model has multiple named outputs
        return predictions['reconstruction_output'] # Return only the reconstruction part





