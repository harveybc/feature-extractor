import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Concatenate, Layer, GRU, Lambda # Added Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import json

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
        self.kl_layer_instance_obj = None # To store the KLDivergenceLayer object
        tf.print(f"[AutoencoderManager] Initialized for CVAE components.")

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
            z_mean, z_log_var = self.encoder_plugin.inference_network_model(
                [cvae_input_x_window, cvae_input_h_context, cvae_input_conditions_t]
            )
            
            kl_beta_start_from_config = config.get('kl_beta_start', 0.0001)
            self.kl_layer_instance_obj = KLDivergenceLayer(kl_beta_start=kl_beta_start_from_config, name="kl_loss_adder_node")
            # KLDivergenceLayer now returns multiple outputs
            kl_processed_z_mean, kl_raw_output, kl_weighted_output, kl_beta_output = \
                self.kl_layer_instance_obj([z_mean, z_log_var])
            
            # --- MODIFIED SAMPLING LOGIC ---
            # Wrap the reparameterization trick in a Lambda layer
            def sampling(args):
                z_mean_sampling, z_log_var_sampling = args
                batch = tf.shape(z_mean_sampling)[0]
                dim = tf.shape(z_mean_sampling)[1]
                # Use keras.ops.random.normal for Keras 3 compatibility if needed,
                # but tf.random.normal should work if tf.shape is handled correctly within Lambda.
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean_sampling + tf.exp(0.5 * z_log_var_sampling) * epsilon

            # The KLDivergenceLayer returns z_mean, so use kl_processed_z_mean and the original z_log_var
            z_sampled = Lambda(sampling, name='z_sampling_lambda')([kl_processed_z_mean, z_log_var])
            # --- END MODIFIED SAMPLING LOGIC ---

            # Decoder plugin's model outputs reconstruction
            reconstruction_output = self.decoder_plugin.generative_network_model(
                [z_sampled, cvae_input_h_context, cvae_input_conditions_t]
            )

            # Give explicit, unique names to output tensors for clarity in compile.
            # These names will be used as keys in the outputs dictionary.
            # Keras might still alter them if they conflict with layer names, so we'll check model.output_names.
            output_dict = {
                'reconstruction_target': reconstruction_output, # Changed key for clarity
                'z_mean_latent': z_mean,                   # Changed key
                'z_log_var_latent': z_log_var,             # Changed key
                'kl_divergence_raw': kl_raw_output,        # Changed key
                'kl_divergence_weighted': kl_weighted_output, # Changed key
                'kl_beta_value': kl_beta_output            # Changed key
            }

            self.autoencoder_model = Model(
                inputs=[cvae_input_x_window, cvae_input_h_context, cvae_input_conditions_t],
                outputs=output_dict, # Use the dictionary with new keys
                name=f"windowed_input_cvae_{num_features_output}_features_out"
            )
            self.model = self.autoencoder_model 

            adam_optimizer = Adam(
                learning_rate=config.get('learning_rate', 0.0001),
                beta_1=config.get('adam_beta_1', 0.9),
                beta_2=config.get('adam_beta_2', 0.999),
                epsilon=config.get('adam_epsilon', 1e-07)
            )
            
            configured_loss_fn = get_reconstruction_and_stats_loss_fn(config) 
            reconstruction_metrics = get_metrics(config=config) 

            tf.print(f"DEBUG: autoencoder_model.outputs (symbolic tensors): {self.autoencoder_model.outputs}")
            tf.print(f"DEBUG: autoencoder_model.output_names (actual names used by Keras): {self.autoencoder_model.output_names}")

            # Define pass-through metric functions
            def pass_through_metric(y_true, y_pred): return y_pred
            
            # --- Dynamically build compile dictionaries based on actual model output names ---
            actual_output_names = self.autoencoder_model.output_names
            
            # Map our desired keys to actual Keras output names
            # This is crucial if Keras renames outputs.
            # For simplicity, we'll assume Keras uses the keys from our `output_dict` if they are valid identifiers
            # and don't clash. If it renames them, this mapping needs to be more robust,
            # potentially by finding the tensor in model.outputs and getting its name.

            # Let's assume the keys we used in `output_dict` are the ones Keras will try to use.
            # The error message suggests Keras is looking for 'reconstruction_output' from your previous `metrics` dict.
            # The `output_names` from your debug log were:
            # ['kl_loss_adder_node', 'kl_loss_adder_node', 'kl_loss_adder_node', 
            #  'dynamic_conv_transpose_cvae_decoder', 
            #  'configurable_conv_bilstm_cvae_encoder', 'configurable_conv_bilstm_cvae_encoder']
            # We need to map these to our conceptual outputs.

            # Tentative mapping based on typical Keras behavior and your layer names:
            # This is an educated guess and might need refinement based on exact Keras naming.
            # It's better if the output tensors themselves are named before creating the Model.
            
            # Let's try to name the tensors directly when creating them or ensure layers have unique names.
            # For now, we will use the keys from our `output_dict` and hope Keras respects them.
            # If not, the error will persist but might point to the correct Keras-generated names.

            # The most important output for loss and metrics:
            recon_output_actual_name = actual_output_names[actual_output_names.index(self.decoder_plugin.generative_network_model.name)] \
                                       if self.decoder_plugin.generative_network_model.name in actual_output_names \
                                       else 'reconstruction_target' # Fallback to our desired name

            # Find the actual names for z_mean and z_log_var (likely from encoder model name)
            # This is tricky if the encoder outputs two tensors. Keras might append _1, _2 or use layer names.
            # Assuming encoder model name is unique and Keras uses it for its outputs:
            encoder_outputs_actual_names = [name for name in actual_output_names if name == self.encoder_plugin.inference_network_model.name]
            z_mean_actual_name = encoder_outputs_actual_names[0] if len(encoder_outputs_actual_names) > 0 else 'z_mean_latent' # Fallback
            z_log_var_actual_name = encoder_outputs_actual_names[1] if len(encoder_outputs_actual_names) > 1 else 'z_log_var_latent' # Fallback

            # Find actual names for KL metrics (likely from KLDivergenceLayer name)
            kl_layer_name = self.kl_layer_instance_obj.name
            kl_outputs_actual_names = [name for name in actual_output_names if name == kl_layer_name]
            # KLDivergenceLayer returns z_mean, kl_raw, kl_weighted, kl_beta.
            # The first output (z_mean) is kl_processed_z_mean, not directly in model outputs dict with this name.
            # The other three are what we named:
            kl_raw_actual_name = kl_outputs_actual_names[1] if len(kl_outputs_actual_names) > 1 else 'kl_divergence_raw' # Fallback (index 1 for 2nd output)
            kl_weighted_actual_name = kl_outputs_actual_names[2] if len(kl_outputs_actual_names) > 2 else 'kl_divergence_weighted' # Fallback
            kl_beta_actual_name = kl_outputs_actual_names[3] if len(kl_outputs_actual_names) > 3 else 'kl_beta_value' # Fallback


            tf.print(f"INFO: Attempting to use actual output name for reconstruction: {recon_output_actual_name}")
            tf.print(f"INFO: Attempting to use actual output name for z_mean: {z_mean_actual_name}")
            tf.print(f"INFO: Attempting to use actual output name for z_log_var: {z_log_var_actual_name}")
            tf.print(f"INFO: Attempting to use actual output name for kl_raw: {kl_raw_actual_name}")


            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss={
                    recon_output_actual_name: configured_loss_fn, 
                    z_mean_actual_name: None, 
                    z_log_var_actual_name: None,
                    kl_raw_actual_name: None, 
                    kl_weighted_actual_name: None,
                    kl_beta_actual_name: None
                },
                loss_weights={ 
                    recon_output_actual_name: 1.0,
                    z_mean_actual_name: 0.0,
                    z_log_var_actual_name: 0.0,
                    kl_raw_actual_name: 0.0,
                    kl_weighted_actual_name: 0.0,
                    kl_beta_actual_name: 0.0
                },
                metrics={ 
                    recon_output_actual_name: reconstruction_metrics, # This should be a list of metric functions
                    kl_raw_actual_name: pass_through_metric, 
                    kl_weighted_actual_name: pass_through_metric,
                    kl_beta_actual_name: pass_through_metric
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
        
        train_inputs_dict = {
            'cvae_input_x_window': data_inputs[0],
            'cvae_input_h_context': data_inputs[1],
            'cvae_input_conditions_t': data_inputs[2]
        }
        
        # Determine the actual name used by Keras for the reconstruction output
        # This should match the key used in model.compile()
        # For robustness, find it from model.output_names
        recon_output_actual_name_train = self.autoencoder_model.output_names[self.autoencoder_model.output_names.index(self.decoder_plugin.generative_network_model.name)] \
                                       if self.decoder_plugin.generative_network_model.name in self.autoencoder_model.output_names \
                                       else 'reconstruction_target' # Fallback to our desired name

        train_targets_dict = {
            recon_output_actual_name_train: data_targets,
        }

        tf.print(f"Input data shapes: x_window: {data_inputs[0].shape}, h_context: {data_inputs[1].shape}, conditions_t: {data_inputs[2].shape}")
        tf.print(f"Target data shape for reconstruction_output: {data_targets.shape}")

        callbacks_list = []
        if config.get('early_stopping_patience', 0) > 0:
            callbacks_list.append(EarlyStoppingWithPatienceCounter(
                monitor=config.get('early_stopping_monitor', 'val_loss'),
                patience=config.get('early_stopping_patience'),
                verbose=1,
                restore_best_weights=config.get('early_stopping_restore_best_weights', True)
            ))

        if config.get('reduce_lr_patience', 0) > 0:
            callbacks_list.append(ReduceLROnPlateauWithCounter(
                monitor=config.get('reduce_lr_monitor', 'val_loss'),
                factor=config.get('reduce_lr_factor', 0.2),
                patience=config.get('reduce_lr_patience'),
                min_lr=config.get('reduce_lr_min_lr', 0.00001),
                verbose=1
            ))
        
        if self.kl_layer_instance_obj and config.get('kl_anneal_epochs', 0) > 0:
            kl_annealing = KLAnnealingCallback(
                kl_beta_start=config.get('kl_beta_start', 0.0001), 
                kl_beta_end=config.get('kl_beta', 1.0), 
                anneal_epochs=config.get('kl_anneal_epochs'),
                kl_layer_instance=self.kl_layer_instance_obj, 
                verbose=1
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
            val_targets_dict = {recon_output_actual_name_train: config['cvae_val_targets']} # Use the same actual name
            validation_data_prepared = (val_inputs_dict, val_targets_dict)
            tf.print(f"[train_autoencoder] Using pre-defined validation data with target key: {recon_output_actual_name_train}")
        else:
            tf.print("[train_autoencoder] No pre-defined validation_data.")

        history = self.autoencoder_model.fit(
            train_inputs_dict,
            train_targets_dict,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            validation_data=validation_data_prepared,
            shuffle=True,
            verbose=config.get('keras_verbose_level', 1)
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

    def load_model(self, model_path, custom_objects=None):
        tf.print(f"Loading CVAE model from {model_path}")
        final_custom_objects = {'KLDivergenceLayer': KLDivergenceLayer}
        if custom_objects:
            final_custom_objects.update(custom_objects)
        
        self.autoencoder_model = load_model(model_path, custom_objects=final_custom_objects, compile=False) 
        self.model = self.autoencoder_model
        
        tf.print(f"Model {self.autoencoder_model.name} loaded. Re-compile if necessary.")
        
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
            tf.print("Model not built or loaded. Cannot evaluate.")
            return None
        
        tf.print(f"Evaluating model on {dataset_name} data...")
        eval_inputs_dict = {
            'cvae_input_x_window': data_inputs[0],
            'cvae_input_h_context': data_inputs[1],
            'cvae_input_conditions_t': data_inputs[2]
        }
        
        recon_output_actual_name_eval = self.autoencoder_model.output_names[self.autoencoder_model.output_names.index(self.decoder_plugin.generative_network_model.name)] \
                                       if self.decoder_plugin.generative_network_model.name in self.autoencoder_model.output_names \
                                       else 'reconstruction_target' # Fallback

        eval_targets_dict = {
            recon_output_actual_name_eval: data_targets
        }
        
        # Store metric names from the compiled model to match with results
        metric_names = self.autoencoder_model.metrics_names
        
        results_list = self.autoencoder_model.evaluate(
            eval_inputs_dict,
            eval_targets_dict,
            batch_size=config.get('batch_size', 128) if config else 128,
            verbose=config.get('keras_verbose_level', 1) if config else 1,
            return_dict=False # Ensure results are a list
        )
        
        results_dict = dict(zip(metric_names, results_list))

        tf.print(f"Evaluation results for {dataset_name} (as dict): {results_dict}")
        return results_dict # Return dictionary for easier access by name

    def predict(self, data_inputs, config=None):
        if not self.autoencoder_model:
            tf.print("Model not built or loaded. Cannot predict.")
            return None
            
        tf.print("Generating predictions...")
        pred_inputs_dict = {
            'cvae_input_x_window': data_inputs[0],
            'cmae_input_h_context': data_inputs[1], # Typo: cmae_ -> cvae_
            'cvae_input_conditions_t': data_inputs[2]
        }
        
        predictions = self.autoencoder_model.predict(
            pred_inputs_dict,
            batch_size=config.get('batch_size', 128) if config else 128,
            verbose=config.get('keras_verbose_level', 1) if config else 1
        )
        
        # Predictions is a dict if model has multiple named outputs
        if isinstance(predictions, dict):
            # Try to find the reconstruction output using the decoder's model name as a key
            recon_output_actual_name_pred = self.autoencoder_model.output_names[self.autoencoder_model.output_names.index(self.decoder_plugin.generative_network_model.name)] \
                                       if self.decoder_plugin.generative_network_model.name in self.autoencoder_model.output_names \
                                       else 'reconstruction_target' # Fallback

            if recon_output_actual_name_pred in predictions:
                return predictions[recon_output_actual_name_pred]
            else: 
                tf.print(f"Warning: Actual reconstruction output name '{recon_output_actual_name_pred}' not in prediction keys {list(predictions.keys())}. Returning first available output.")
                return list(predictions.values())[0]
        return predictions # If single unnamed output





