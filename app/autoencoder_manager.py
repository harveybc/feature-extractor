import tensorflow as tf
from tensorflow.keras.models import Model, load_model # Changed
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Concatenate, Layer, GRU, Lambda # Changed
from tensorflow.keras.optimizers import Adam # Changed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Changed
import numpy as np
import os
import json

from app.autoencoder_helper import (
    get_reconstruction_and_stats_loss_fn, 
    get_metrics, # Ensure this is imported and used
    EarlyStoppingWithPatienceCounter,
    ReduceLROnPlateauWithCounter,
    KLAnnealingCallback,
    EpochEndLogger 
)

class KLDivergenceLayer(tf.keras.layers.Layer): # Changed
    def __init__(self, kl_beta_start=0.001, name="kl_loss_adder_node", **kwargs):
        super(KLDivergenceLayer, self).__init__(name=name, **kwargs)
        self.kl_beta_val = float(kl_beta_start) 
        self.kl_beta = tf.Variable(self.kl_beta_val, trainable=False, dtype=tf.float32, name="kl_beta_weight")
        tf.print(f"KLDivergenceLayer '{self.name}' initialized with kl_beta_start: {self.kl_beta_val}")

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # tf.reduce_sum(..., axis=1) reduces over features, result shape (batch_size,)
        # tf.reduce_mean(...) then averages over the batch, resulting in a scalar.
        kl_loss_raw = -0.5 * tf.reduce_mean( 
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        # The check 'if tf.rank(kl_loss_raw) != 0:' is removed as kl_loss_raw should be scalar.
        # If it wasn't, the tf.reduce_mean above would make it so.

        weighted_kl_loss = self.kl_beta * kl_loss_raw
        self.add_loss(weighted_kl_loss) # This adds the loss to the model
        
        return z_mean, kl_loss_raw, weighted_kl_loss, self.kl_beta 

    def compute_output_shape(self, input_shape):
        # input_shape is a list: [z_mean_shape, z_log_var_shape]
        z_mean_shape = tf.TensorShape(input_shape[0])
        # kl_loss_raw, weighted_kl_loss, and self.kl_beta are scalars
        scalar_shape = tf.TensorShape([])
        return [z_mean_shape, scalar_shape, scalar_shape, scalar_shape]

    def get_config(self):
        config = super().get_config()
        config.update({"kl_beta_start": float(self.kl_beta.numpy())}) 
        return config

class AutoencoderManager:
    def __init__(self, encoder_plugin, decoder_plugin):
        self.encoder_plugin = encoder_plugin
        self.decoder_plugin = decoder_plugin
        self.autoencoder_model = None
        self.model = None 
        self.kl_layer_instance_obj = None 
        tf.print(f"[AutoencoderManager] Initialized for CVAE components.")

    def _compile_model(self, config):
        """Helper method to compile the autoencoder_model."""
        if not self.autoencoder_model:
            tf.print("Error: Model not available in _compile_model. Cannot compile.")
            raise RuntimeError("Model not available for compilation in _compile_model.")

        adam_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', 0.0001),
            beta_1=config.get('adam_beta_1', 0.9),
            beta_2=config.get('adam_beta_2', 0.999),
            epsilon=config.get('adam_epsilon', 1e-07)
        )
        
        configured_loss_fn = get_reconstruction_and_stats_loss_fn(config) 
        
        helper_mae_functions = get_metrics(config) # config is optional for current get_metrics
        if not helper_mae_functions:
            raise ValueError("get_metrics(config) from autoencoder_helper returned an empty list or None.")
        # We'll use a standard Keras MAE instance for the dedicated output
        # mae_fn_for_compile = helper_mae_functions[0] 
        # tf.print(f"DEBUG: MAE function available from get_metrics(): {mae_fn_for_compile.__name__}")

        tf.print(f"DEBUG: Compiling with dedicated output 'reconstruction_out_for_mae_calc' for MAE.")
        tf.print(f"DEBUG: Compiling model. Output names for compile: {self.autoencoder_model.output_names}")

        def pass_through_metric(y_true, y_pred): return y_pred 

        metrics_dict_for_compile = { 
            # No explicit MAE metric on 'reconstruction_out' if it's causing issues with custom loss
            # 'reconstruction_out': [tf.keras.metrics.MeanAbsoluteError()], # REMOVED/COMMENTED
            'reconstruction_out_for_mae_calc': [tf.keras.metrics.MeanAbsoluteError()], # MAE on dedicated output
            'kl_raw_out': pass_through_metric, 
            'kl_weighted_out': pass_through_metric,
            'kl_beta_out': pass_through_metric 
        }
        tf.print(f"DEBUG: Metrics dictionary for compile: {metrics_dict_for_compile}")

        self.autoencoder_model.compile(
            optimizer=adam_optimizer,
            loss={
                'reconstruction_out': configured_loss_fn, 
                'reconstruction_out_for_mae_calc': None, # IMPORTANT: Loss is None for MAE-dedicated output
                'z_mean_out': None, 
                'z_log_var_out': None,
                'kl_raw_out': None, 
                'kl_weighted_out': None,
                'kl_beta_out': None
            },
            loss_weights={ 
                'reconstruction_out': 1.0,
                'reconstruction_out_for_mae_calc': 0.0, # IMPORTANT: No contribution to overall loss
                'z_mean_out': 0.0,
                'z_log_var_out': 0.0,
                'kl_raw_out': 0.0,
                'kl_weighted_out': 0.0,
                'kl_beta_out': 0.0
            },
            metrics=metrics_dict_for_compile,
            run_eagerly=config.get('run_eagerly', False) 
        )
        tf.print("[_compile_model] Model compiled successfully.")
        tf.print(f"[_compile_model] Model metrics_names AFTER compile: {self.autoencoder_model.metrics_names}")
        tf.print(f"[_compile_model] Model.metrics (metric objects) AFTER compile: Count: {len(self.autoencoder_model.metrics)}")
        for i, metric_obj in enumerate(self.autoencoder_model.metrics):
            try:
                config_str = str(metric_obj.get_config()) if hasattr(metric_obj, 'get_config') else "N/A"
                tf.print(f"[_compile_model] Metric object {i}: Name: {metric_obj.name}, Class: {metric_obj.__class__.__name__}, Dtype: {metric_obj.dtype}, Config: {config_str}")
            except Exception as e:
                tf.print(f"[_compile_model] Metric object {i}: Name: {metric_obj.name}, Class: {metric_obj.__class__.__name__}, Error inspecting: {e}")


    def build_autoencoder(self, config):
        try:
            tf.print("[build_autoencoder] Starting to build single-step CVAE...")
            
            window_size = config['window_size']
            num_features_input = config['num_features_input']
            rnn_hidden_dim = config['rnn_hidden_dim'] 
            conditioning_dim = config['conditioning_dim'] 
            latent_dim = config['latent_dim']
            num_features_output = config['num_features_output']

            # 1. Configure Encoder Plugin
            self.encoder_plugin.configure_model_architecture(
                window_size=window_size,
                input_features_per_step=num_features_input,
                rnn_hidden_dim=rnn_hidden_dim,
                conditioning_dim=conditioning_dim,
                latent_dim=latent_dim,
                config=config 
            )
            if not hasattr(self.encoder_plugin, 'inference_network_model') or self.encoder_plugin.inference_network_model is None:
                raise RuntimeError("Encoder plugin did not build its internal model (expected attribute 'inference_network_model').")

            # 2. Configure Decoder Plugin
            self.decoder_plugin.configure_model_architecture(
                latent_dim=latent_dim,
                rnn_hidden_dim=rnn_hidden_dim, 
                conditioning_dim=conditioning_dim,
                output_feature_dim=num_features_output, # Decoder plugin will force this to 6
                config=config
            )
            if not hasattr(self.decoder_plugin, 'generative_network_model') or self.decoder_plugin.generative_network_model is None:
                raise RuntimeError("Decoder plugin did not build its internal model (expected attribute 'generative_network_model').")

            cvae_input_x_window = Input(shape=(window_size, num_features_input), name='cvae_input_x_window')
            cvae_input_h_context = Input(shape=(rnn_hidden_dim,), name='cvae_input_h_context') 
            cvae_input_conditions_t = Input(shape=(conditioning_dim,), name='cvae_input_conditions_t')

            # Encoder plugin's model outputs z_mean, z_log_var
            # These are intermediate tensors before explicit naming for model output
            encoder_z_mean, encoder_z_log_var = self.encoder_plugin.inference_network_model(
                [cvae_input_x_window, cvae_input_h_context, cvae_input_conditions_t]
            )
            
            kl_beta_start_from_config = config.get('kl_beta_start', 0.0001)
            self.kl_layer_instance_obj = KLDivergenceLayer(kl_beta_start=kl_beta_start_from_config, name="kl_loss_adder_node")
            # KLDivergenceLayer returns: z_mean (passed through), kl_loss_raw, weighted_kl_loss, kl_beta
            kl_processed_z_mean, inter_kl_raw, inter_kl_weighted, inter_kl_beta = \
                self.kl_layer_instance_obj([encoder_z_mean, encoder_z_log_var]) # Ensure this is called
            
            def sampling(args):
                z_mean_sampling, z_log_var_sampling = args
                batch = tf.shape(z_mean_sampling)[0]
                dim = tf.shape(z_mean_sampling)[1]
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean_sampling + tf.exp(0.5 * z_log_var_sampling) * epsilon

            z_sampled = Lambda(sampling, name='z_sampling_lambda')([kl_processed_z_mean, encoder_z_log_var])

            # Decoder plugin's model outputs reconstruction (intermediate tensor)
            intermediate_reconstruction_output = self.decoder_plugin.generative_network_model(
                [z_sampled, cvae_input_h_context, cvae_input_conditions_t]
            )

            # --- Explicitly name all final output tensors using Lambda layers ---
            final_reconstruction_output = Lambda(lambda x: x, name='reconstruction_out')(intermediate_reconstruction_output)
            # NEW: Duplicate reconstruction output for dedicated MAE calculation
            final_reconstruction_for_mae = Lambda(lambda x: x, name='reconstruction_out_for_mae_calc')(intermediate_reconstruction_output)

            final_z_mean = Lambda(lambda x: x, name='z_mean_out')(encoder_z_mean) # Use original z_mean from encoder
            final_z_log_var = Lambda(lambda x: x, name='z_log_var_out')(encoder_z_log_var) # Use original z_log_var
            final_kl_raw = Lambda(lambda x: x, name='kl_raw_out')(inter_kl_raw)
            final_kl_weighted = Lambda(lambda x: x, name='kl_weighted_out')(inter_kl_weighted)
            final_kl_beta = Lambda(lambda x: x, name='kl_beta_out')(inter_kl_beta)

            outputs_for_model = {
                'reconstruction_out': final_reconstruction_output,
                'reconstruction_out_for_mae_calc': final_reconstruction_for_mae, # ADDED NEW OUTPUT
                'z_mean_out': final_z_mean,
                'z_log_var_out': final_z_log_var,
                'kl_raw_out': final_kl_raw,
                'kl_weighted_out': final_kl_weighted,
                'kl_beta_out': final_kl_beta
            }

            self.autoencoder_model = Model(
                inputs=[cvae_input_x_window, cvae_input_h_context, cvae_input_conditions_t],
                outputs=outputs_for_model,
                name=f"windowed_input_cvae_{num_features_output}_features_out"
            )
            self.model = self.autoencoder_model 

            self._compile_model(config) # Call the compilation helper
            tf.print("[build_autoencoder] Single-step CVAE model built and compiled.")

        except Exception as e:
            tf.print(f"Error during CVAE model building: {e}")
            import traceback
            tf.print(traceback.format_exc())
            raise
    
    def train_autoencoder(self, data_inputs, data_targets, epochs, batch_size, config):
        tf.print("[train_autoencoder] Starting CVAE training.")
        
        if not self.autoencoder_model.optimizer:
            tf.print("[train_autoencoder] Model was not compiled (no optimizer found). Compiling now.")
            self._compile_model(config)
        
        tf.print(f"[train_autoencoder] PRE-FIT CHECK: Model.metrics_names: {self.autoencoder_model.metrics_names}") 
        tf.print(f"[train_autoencoder] PRE-FIT CHECK: Model.metrics (metric objects count): {len(self.autoencoder_model.metrics)}")
        for i, metric_obj in enumerate(self.autoencoder_model.metrics):
            tf.print(f"[train_autoencoder] PRE-FIT CHECK: Metric {i} - Name: {metric_obj.name}, Class: {metric_obj.__class__.__name__}")

        train_inputs_dict = {
            'cvae_input_x_window': data_inputs[0],
            'cvae_input_h_context': data_inputs[1],
            'cvae_input_conditions_t': data_inputs[2]
        }
        
        num_samples_train = data_targets.shape[0]
        latent_dim = config['latent_dim']

        train_targets_dict = {}
        for output_name in self.autoencoder_model.output_names:
            if output_name == 'reconstruction_out':
                train_targets_dict[output_name] = data_targets
            elif output_name == 'reconstruction_out_for_mae_calc':
                train_targets_dict[output_name] = data_targets
            elif output_name in ['z_mean_out', 'z_log_var_out']:
                train_targets_dict[output_name] = np.zeros((num_samples_train, latent_dim), dtype=np.float32)
            elif output_name in ['kl_raw_out', 'kl_weighted_out', 'kl_beta_out']:
                train_targets_dict[output_name] = np.zeros((num_samples_train, 1), dtype=np.float32)
            else: # Should not happen if all outputs are covered
                train_targets_dict[output_name] = np.zeros((num_samples_train, 1), dtype=np.float32) # Default dummy
                tf.print(f"Warning: Unhandled output_name '{output_name}' in train_targets_dict. Using default dummy shape (num_samples, 1).")
        
        tf.print(f"[train_autoencoder] train_targets_dict keys: {list(train_targets_dict.keys())}")
        for k, v in train_targets_dict.items():
            tf.print(f"  Target '{k}' shape: {v.shape if hasattr(v, 'shape') else 'None'}")

        tf.print(f"Input data shapes: x_window: {data_inputs[0].shape}, h_context: {data_inputs[1].shape}, conditions_t: {data_inputs[2].shape}")
        tf.print(f"Target data shape for reconstruction_output: {data_targets.shape}")

        callbacks_list = []
        
        # Add EpochEndLogger: This will print consolidated info at epoch end
        callbacks_list.append(EpochEndLogger())

        if config.get('early_stopping_patience', 0) > 0:
            callbacks_list.append(EarlyStoppingWithPatienceCounter(
                monitor=config.get('early_stopping_monitor', 'val_loss'),
                patience=config.get('early_stopping_patience'),
                verbose=0, # MODIFIED: Set to 0 to let EpochEndLogger handle printing
                restore_best_weights=config.get('early_stopping_restore_best_weights', True)
            ))

        if config.get('reduce_lr_patience', 0) > 0:
            callbacks_list.append(ReduceLROnPlateauWithCounter(
                monitor=config.get('reduce_lr_monitor', 'val_loss'),
                factor=config.get('reduce_lr_factor', 0.2),
                patience=config.get('reduce_lr_patience'),
                min_lr=config.get('reduce_lr_min_lr', 0.00001),
                verbose=0 # MODIFIED: Set to 0 to let EpochEndLogger handle printing
            ))
        
        if self.kl_layer_instance_obj and config.get('kl_anneal_epochs', 0) > 0:
            kl_annealing = KLAnnealingCallback(
                kl_beta_start=config.get('kl_beta_start', 0.0001), 
                kl_beta_end=config.get('kl_beta', 1.0), 
                anneal_epochs=config.get('kl_anneal_epochs'),
                kl_layer_instance=self.kl_layer_instance_obj, 
                verbose=0 # Verbose for KLAnnealing can remain 0 as EpochEndLogger handles its state
            )
            callbacks_list.append(kl_annealing)
            tf.print("[train_autoencoder] KL Annealing callback configured with layer object.")
        elif config.get('kl_anneal_epochs', 0) > 0:
             tf.print("[train_autoencoder] KL Annealing specified in config, but KLDivergenceLayer object not found/stored. Skipping KLAnnealingCallback.")
        else:
            tf.print("[train_autoencoder] KL Annealing not configured (kl_anneal_epochs is 0).")

        validation_data_prepared = None
        if 'cvae_val_inputs' in config and 'cvae_val_targets' in config:
            val_inputs_dict = {
                'cvae_input_x_window': config['cvae_val_inputs']['x_window'],
                'cvae_input_h_context': config['cvae_val_inputs']['h_context'],
                'cvae_input_conditions_t': config['cvae_val_inputs']['conditions_t']
            }
            # Use the explicitly defined name for validation target
            val_targets_dict = {'reconstruction_out': config['cvae_val_targets']}
            validation_data_prepared = (val_inputs_dict, val_targets_dict)
            tf.print(f"[train_autoencoder] Using pre-defined validation data with target key: 'reconstruction_out'")
        else:
            tf.print("[train_autoencoder] No pre-defined validation_data.")

        if validation_data_prepared:
            val_inputs_dict_original, val_targets_dict_original = validation_data_prepared
            
            # Assuming val_targets_dict_original['reconstruction_out'] is the primary validation target array
            num_samples_val = val_targets_dict_original['reconstruction_out'].shape[0]

            val_targets_dict_new = {}
            for output_name in self.autoencoder_model.output_names:
                if output_name == 'reconstruction_out':
                    val_targets_dict_new[output_name] = val_targets_dict_original['reconstruction_out']
                elif output_name == 'reconstruction_out_for_mae_calc':
                    val_targets_dict_new[output_name] = val_targets_dict_original['reconstruction_out']
                elif output_name in ['z_mean_out', 'z_log_var_out']:
                    val_targets_dict_new[output_name] = np.zeros((num_samples_val, latent_dim), dtype=np.float32)
                elif output_name in ['kl_raw_out', 'kl_weighted_out', 'kl_beta_out']:
                    val_targets_dict_new[output_name] = np.zeros((num_samples_val, 1), dtype=np.float32)
                else: # Should not happen
                    val_targets_dict_new[output_name] = np.zeros((num_samples_val, 1), dtype=np.float32)
                    tf.print(f"Warning: Unhandled output_name '{output_name}' in val_targets_dict_new. Using default dummy shape (num_samples, 1).")

            tf.print(f"[train_autoencoder] val_targets_dict_new keys: {list(val_targets_dict_new.keys())}")
            for k, v in val_targets_dict_new.items():
                tf.print(f"  Val Target '{k}' shape: {v.shape if hasattr(v, 'shape') else 'None'}")
            validation_data_prepared = (val_inputs_dict_original, val_targets_dict_new)

        history = self.autoencoder_model.fit(
            train_inputs_dict,
            train_targets_dict,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            validation_data=validation_data_prepared,
            shuffle=True,
            verbose=config.get('keras_verbose_level', 1) # Keras's main verbose for progress bar etc.
        )
        return history

    def save_model(self, model_path, encoder_path=None, decoder_path=None):
        if self.autoencoder_model:
            tf.print(f"Saving full CVAE model to {model_path}")
            self.autoencoder_model.save(model_path) 
            
        if encoder_path and self.encoder_plugin and hasattr(self.encoder_plugin, 'save'):
            tf.print(f"Saving encoder plugin's internal model to {encoder_path}")
            self.encoder_plugin.save(encoder_path) 
        elif encoder_path:
            tf.print(f"Encoder path provided but encoder_plugin not available or has no save method.")

        if decoder_path and self.decoder_plugin and hasattr(self.decoder_plugin, 'save'):
            tf.print(f"Saving decoder plugin's internal model to {decoder_path}")
            self.decoder_plugin.save(decoder_path) 
        elif decoder_path:
            tf.print(f"Decoder path provided but decoder_plugin not available or has no save method.")
        
        if not self.autoencoder_model and \
           not (encoder_path and self.encoder_plugin and hasattr(self.encoder_plugin, 'save')) and \
           not (decoder_path and self.decoder_plugin and hasattr(self.decoder_plugin, 'save')):
            tf.print("No model or plugin components to save.")

    def load_model(self, model_path, config, custom_objects=None): # Added config
        tf.print(f"Loading CVAE model from {model_path}")
        # Ensure KLDivergenceLayer is passed if it's a custom layer in the saved model
        final_custom_objects = {'KLDivergenceLayer': KLDivergenceLayer}
        if custom_objects:
            final_custom_objects.update(custom_objects) # Merge with any other custom objects
        
        self.autoencoder_model = tf.keras.models.load_model( # Changed
            model_path, 
            custom_objects=final_custom_objects, 
            compile=False # Compile will be handled by _compile_model
        ) 
        self.model = self.autoencoder_model
        
        tf.print(f"Model {self.autoencoder_model.name} loaded.")
        
        if config:
            tf.print(f"Re-compiling loaded model with provided config...")
            self._compile_model(config) # Re-compile the loaded model using the helper
        else:
            tf.print("Warning: Model loaded but no config provided for re-compilation. Evaluate/train might use defaults or fail if not compiled elsewhere.")

        try:
            self.kl_layer_instance_obj = self.autoencoder_model.get_layer("kl_loss_adder_node")
            tf.print(f"KL Divergence layer '{self.kl_layer_instance_obj.name}' found in loaded model.")
        except ValueError:
            self.kl_layer_instance_obj = None
            tf.print("KL Divergence layer 'kl_loss_adder_node' not found in loaded model.")

    def save_encoder(self, encoder_path):
        if self.encoder_plugin and hasattr(self.encoder_plugin, 'save'):
            tf.print(f"Saving encoder plugin's internal model via AutoencoderManager to {encoder_path}")
            self.encoder_plugin.save(encoder_path)
        else:
            tf.print("Encoder plugin not available or does not have a save method.")

    def save_decoder(self, decoder_path):
        if self.decoder_plugin and hasattr(self.decoder_plugin, 'save'):
            tf.print(f"Saving decoder plugin's internal model via AutoencoderManager to {decoder_path}")
            self.decoder_plugin.save(decoder_path)
        else:
            tf.print("Decoder plugin not available or does not have a save method.")

    def evaluate(self, data_inputs, data_targets, dataset_name="Test", config=None):
        if not self.autoencoder_model:
            print("Model not built or loaded. Cannot evaluate.")
            return None

        if not self.autoencoder_model.optimizer:
            print(f"Warning: Model for evaluation on '{dataset_name}' was not compiled. Attempting to compile.")
            if config:
                self._compile_model(config) 
            else:
                print("Error: Cannot compile model for evaluation as no config was provided.")
                return None
        
        tf.print(f"[evaluate] PRE-EVALUATE CHECK on '{dataset_name}': Model.metrics_names: {self.autoencoder_model.metrics_names}") 
        tf.print(f"[evaluate] PRE-EVALUATE CHECK on '{dataset_name}': Model.metrics (metric objects count): {len(self.autoencoder_model.metrics)}")
        for i, metric_obj in enumerate(self.autoencoder_model.metrics):
            tf.print(f"[evaluate] PRE-EVALUATE CHECK on '{dataset_name}': Metric {i} - Name: {metric_obj.name}, Class: {metric_obj.__class__.__name__}")
        
        print(f"Evaluating model on {dataset_name} data...")
        eval_inputs_dict = {
            'cvae_input_x_window': data_inputs[0],
            'cvae_input_h_context': data_inputs[1],
            'cvae_input_conditions_t': data_inputs[2]
        }
        
        num_samples_eval = data_targets.shape[0]
        latent_dim = config['latent_dim']

        eval_targets_dict = {}
        for output_name in self.autoencoder_model.output_names:
            if output_name == 'reconstruction_out':
                eval_targets_dict[output_name] = data_targets
            elif output_name == 'reconstruction_out_for_mae_calc':
                eval_targets_dict[output_name] = data_targets
            elif output_name in ['z_mean_out', 'z_log_var_out']:
                eval_targets_dict[output_name] = np.zeros((num_samples_eval, latent_dim), dtype=np.float32)
            elif output_name in ['kl_raw_out', 'kl_weighted_out', 'kl_beta_out']:
                eval_targets_dict[output_name] = np.zeros((num_samples_eval, 1), dtype=np.float32)
            else: # Should not happen
                eval_targets_dict[output_name] = np.zeros((num_samples_eval, 1), dtype=np.float32)
                tf.print(f"Warning: Unhandled output_name '{output_name}' in eval_targets_dict. Using default dummy shape (num_samples, 1).")

        tf.print(f"[evaluate] eval_targets_dict keys for '{dataset_name}': {list(eval_targets_dict.keys())}")
        for k, v in eval_targets_dict.items():
            tf.print(f"  Eval Target '{k}' shape: {v.shape if hasattr(v, 'shape') else 'None'}")

        tf.print(f"[evaluate] Shapes for '{dataset_name}':")
        tf.print(f"  cvae_input_x_window: {tf.shape(data_inputs[0])}")
        tf.print(f"  cvae_input_h_context: {tf.shape(data_inputs[1])}")
        tf.print(f"  cvae_input_conditions_t: {tf.shape(data_inputs[2])}")
        tf.print(f"  reconstruction_out (targets): {tf.shape(data_targets)}")
        
        # Print a few sample values from targets for evaluation
        tf.print(f"  reconstruction_out (targets) sample (first 5 of first batch):", data_targets[0,:5] if tf.shape(data_targets)[0] > 0 else "N/A")
        # Check for NaNs/Infs in evaluation targets
        tf.print(f"  Any NaNs in '{dataset_name}' targets (reconstruction_out):", tf.reduce_any(tf.math.is_nan(tf.cast(data_targets, tf.float32))))
        tf.print(f"  Any Infs in '{dataset_name}' targets (reconstruction_out):", tf.reduce_any(tf.math.is_inf(tf.cast(data_targets, tf.float32))))

        results_dict = self.autoencoder_model.evaluate(
            eval_inputs_dict,
            eval_targets_dict,
            batch_size=config.get('batch_size', 128) if config else 128,
            verbose=0, 
            return_dict=True 
        )
        
        print(f"Evaluation results for {dataset_name} (from Keras return_dict=True): {results_dict}")
        tf.print(f"[evaluate DEBUG] ALL KEYS in results_dict for '{dataset_name}': {list(results_dict.keys())}") 
        for key, value in results_dict.items(): 
            tf.print(f"[evaluate DEBUG]    '{key}': {value} (type: {type(value)})")
        return results_dict 

    def predict(self, data_inputs, config=None):
        if not self.autoencoder_model:
            tf.print("Model not built or loaded. Cannot predict.")
            return None
            
        tf.print("Generating predictions...")
        pred_inputs_dict = {
            'cvae_input_x_window': data_inputs[0],
            'cvae_input_h_context': data_inputs[1], # Corrected typo from cmae_
            'cvae_input_conditions_t': data_inputs[2]
        }
        
        predictions = self.autoencoder_model.predict(
            pred_inputs_dict,
            batch_size=config.get('batch_size', 128) if config else 128,
            verbose=config.get('keras_verbose_level', 1) if config else 1
        )
        
        # Predictions is a dict if model has multiple named outputs
        if isinstance(predictions, dict):
            # Use the explicitly defined name to retrieve predictions
            if 'reconstruction_out' in predictions:
                return predictions['reconstruction_out']
            else: 
                tf.print(f"Warning: Explicitly named 'reconstruction_out' not in prediction keys {list(predictions.keys())}. Returning first available output.")
                return list(predictions.values())[0]
        return predictions # If single unnamed output





