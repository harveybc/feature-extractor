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
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        weighted_kl_loss = self.kl_beta * kl_loss
        self.add_loss(weighted_kl_loss)
        
        self.add_metric(kl_loss, name="kl_divergence_raw") 
        self.add_metric(weighted_kl_loss, name="kl_divergence_weighted") 
        self.add_metric(self.kl_beta, name="kl_beta_current")
        return z_mean 

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 0 and isinstance(input_shape[0], tuple):
            return input_shape[0]
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and \
           isinstance(input_shape[0], (list, tuple, tf.TensorShape)):
             return tf.TensorShape(input_shape[0])
        if isinstance(input_shape, tf.TensorShape):
            return input_shape
        tf.print(f"Warning: KLDivergenceLayer.compute_output_shape received unexpected input_shape: {input_shape}. Falling back.")
        # Fallback, assuming the first element of a list is the shape of z_mean
        if isinstance(input_shape, list) and len(input_shape) > 0:
            return input_shape[0]
        raise ValueError(f"Unexpected input_shape format for KLDivergenceLayer: {input_shape}")

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
            kl_processed_z_mean = self.kl_layer_instance_obj([z_mean, z_log_var])
            
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
            # It expects [z_sampled, h_context, conditions_t]
            reconstruction_output = self.decoder_plugin.generative_network_model(
                [z_sampled, cvae_input_h_context, cvae_input_conditions_t]
            )

            self.autoencoder_model = Model(
                inputs=[cvae_input_x_window, cvae_input_h_context, cvae_input_conditions_t],
                outputs={
                    'reconstruction_output': reconstruction_output,
                    'z_mean_output': z_mean,       
                    'z_log_var_output': z_log_var  
                },
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

            tf.print(f"DEBUG: autoencoder_model.outputs: {self.autoencoder_model.outputs}")
            tf.print(f"DEBUG: autoencoder_model.output_names: {self.autoencoder_model.output_names}")

            self.autoencoder_model.compile(
                optimizer=adam_optimizer,
                loss={
                    'reconstruction_output': configured_loss_fn, 
                    'z_mean_output': None, 
                    'z_log_var_output': None
                },
                loss_weights={ 
                    'reconstruction_output': 1.0,
                    'z_mean_output': 0.0,
                    'z_log_var_output': 0.0
                },
                metrics={ 
                    'reconstruction_output': reconstruction_metrics
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
        train_targets_dict = {'reconstruction_output': data_targets}

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
            val_targets_dict = {'reconstruction_output': config['cvae_val_targets']}
            validation_data_prepared = (val_inputs_dict, val_targets_dict)
            tf.print(f"[train_autoencoder] Using pre-defined validation data.")
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
        eval_targets_dict = {'reconstruction_output': data_targets}
        
        results = self.autoencoder_model.evaluate(
            eval_inputs_dict,
            eval_targets_dict,
            batch_size=config.get('batch_size', 128) if config else 128,
            verbose=config.get('keras_verbose_level', 1) if config else 1
        )
        
        tf.print(f"Evaluation results for {dataset_name}:")
        if isinstance(results, list): 
            i = 0
            tf.print(f"  Total Loss: {results[i]}")
            i += 1
            # Iterate through compiled metrics which include losses for outputs and other metrics
            metric_idx_start = 1 # Start after total loss
            for output_name in self.autoencoder_model.output_names:
                # Find the loss for this output if it exists (not None in compile)
                loss_fn_for_output_idx = self.autoencoder_model.output_names.index(output_name)
                if self.autoencoder_model.loss_functions[loss_fn_for_output_idx] is not None:
                    loss_name_in_metrics = f"{output_name}_loss" # Keras default naming
                    # Check if this loss name is in metrics_names (it might not be if loss_weight is 0)
                    # Or directly use the order if Keras guarantees it.
                    # For simplicity, let's assume Keras lists output losses first if they contribute.
                    # A more robust way is to map self.autoencoder_model.metrics_names
                    tf.print(f"  Loss for '{output_name}': {results[metric_idx_start]}") # This assumes order
                    metric_idx_start +=1
                
                # Find metrics associated with this output
                # This part is tricky as Keras flattens metrics_names
                # For now, let's print all remaining metrics by name
            
            # Print all metrics by name from metrics_names
            for metric_name_idx, name in enumerate(self.autoencoder_model.metrics_names):
                if name == "loss": continue # Already printed as Total Loss
                if name.endswith("_loss") and name.split("_loss")[0] in self.autoencoder_model.output_names:
                    # This was an output-specific loss, already handled if it contributed
                    # Or, if it didn't contribute (weight 0), it might still be here.
                    # For now, we assume the previous loop handled contributing losses.
                    continue 
                tf.print(f"  Metric '{name}': {results[metric_name_idx]}") # metric_name_idx is for the full results list

        else: 
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
        
        # Predictions is a dict if model has multiple named outputs
        if isinstance(predictions, dict):
            if 'reconstruction_output' in predictions:
                return predictions['reconstruction_output']
            else: # Fallback if the primary output name is different or only one output
                tf.print(f"Warning: 'reconstruction_output' not in prediction keys. Returning first available output.")
                return list(predictions.values())[0]
        return predictions # If single unnamed output





