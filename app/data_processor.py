import numpy as np
import pandas as pd # Added for DataFrame creation in evaluate functions
import time # Added for execution_time
import tensorflow as tf # Added for tf.print consistency
from app.autoencoder_manager import AutoencoderManager
from app.data_handler import load_csv, write_csv
# from app.reconstruction import unwindow_data 
from app.config_handler import save_debug_info, remote_log
from keras.models import load_model

# This utility function might still be useful for a preprocessor plugin,
# but not directly for creating the CVAE's per-step inputs in this file.
def create_sliding_windows(data, window_size):
    """
    Create sliding windows for the entire dataset, returning a 3D array.
    
    Args:
        data (pd.DataFrame): Dataframe containing the time series features.
        window_size (int): The length of the sliding window.
    
    Returns:
        np.ndarray: Array with shape (num_samples, window_size, num_features).
    """
    data_array = data.to_numpy()
    num_features = data_array.shape[1]
    num_samples = data_array.shape[0] - window_size + 1
    
    windows = np.zeros((num_samples, window_size, num_features))
    
    for i in range(num_samples):
        windows[i] = data_array[i:i+window_size]
    
    return windows


def run_autoencoder_pipeline(config, encoder_plugin, decoder_plugin, preprocessor_plugin):
    start_time = time.time()
    
    tf.print("Loading/processing datasets via PreprocessorPlugin...")
    datasets = preprocessor_plugin.run_preprocessing(config)
    tf.print("PreprocessorPlugin finished.")

    x_train_data = datasets.get("x_train") 
    x_val_data = datasets.get("x_val")
    feature_names_all = datasets.get("feature_names") 

    if x_train_data is None: # x_val_data can be None if no validation is performed
        raise ValueError("PreprocessorPlugin did not return 'x_train' data.")
    if feature_names_all is None:
        raise ValueError("PreprocessorPlugin did not return 'feature_names'.")
    if not isinstance(feature_names_all, list):
        raise TypeError(f"'feature_names' must be a list, got {type(feature_names_all)}")

    target_feature_names = config.get('cvae_target_feature_names', ['OPEN', 'LOW', 'HIGH', 'vix_close', 'BC-BO', 'BH-BL'])
    tf.print(f"Using CVAE target features: {target_feature_names}")
    
    try:
        target_indices = [feature_names_all.index(name) for name in target_feature_names]
    except ValueError as e:
        missing_feature = str(e).split("'")[1] 
        raise ValueError(
            f"One of the CVAE target features ('{missing_feature}') not found in 'feature_names' "
            f"provided by PreprocessorPlugin. Available features: {feature_names_all}"
        ) from e

    if len(x_train_data.shape) != 3:
        raise ValueError(f"x_train_data is expected to be 3D (samples, window, features), but got shape {x_train_data.shape}")
    
    y_train_targets_6_features = x_train_data[:, -1, target_indices]
    tf.print(f"Constructed y_train_targets_6_features with shape: {y_train_targets_6_features.shape}")

    y_val_targets_6_features = None
    if x_val_data is not None:
        if len(x_val_data.shape) != 3:
            raise ValueError(f"x_val_data is expected to be 3D (samples, window, features), but got shape {x_val_data.shape}")
        y_val_targets_6_features = x_val_data[:, -1, target_indices]
        tf.print(f"Constructed y_val_targets_6_features with shape: {y_val_targets_6_features.shape}")
        if x_val_data.shape[0] != y_val_targets_6_features.shape[0]:
            raise ValueError(f"Sample count mismatch between x_val_data ({x_val_data.shape[0]}) and y_val_targets_6_features ({y_val_targets_6_features.shape[0]})")
    else:
        tf.print("No x_val_data provided by preprocessor. Validation will be skipped if not set in config later.")


    if not isinstance(y_train_targets_6_features, np.ndarray):
        raise TypeError(f"y_train_targets_6_features is type {type(y_train_targets_6_features)}, expected np.ndarray.")
    if y_val_targets_6_features is not None and not isinstance(y_val_targets_6_features, np.ndarray):
        raise TypeError(f"y_val_targets_6_features is type {type(y_val_targets_6_features)}, expected np.ndarray.")

    if y_train_targets_6_features.shape[-1] != len(target_feature_names) or (len(y_train_targets_6_features.shape) != 2):
        raise ValueError(f"y_train_targets_6_features should be 2D with {len(target_feature_names)} features, but got shape {y_train_targets_6_features.shape}")
    if y_val_targets_6_features is not None and (y_val_targets_6_features.shape[-1] != len(target_feature_names) or (len(y_val_targets_6_features.shape) != 2)):
        raise ValueError(f"y_val_targets_6_features should be 2D with {len(target_feature_names)} features, but got shape {y_val_targets_6_features.shape}")
    
    if x_train_data.shape[0] != y_train_targets_6_features.shape[0]:
        raise ValueError(f"Sample count mismatch between x_train_data ({x_train_data.shape[0]}) and y_train_targets_6_features ({y_train_targets_6_features.shape[0]})")

    # --- Populate necessary dimensions in config ---
    if 'window_size' not in config or not isinstance(config.get('window_size'), int) or config.get('window_size') <= 0:
        config['window_size'] = x_train_data.shape[1]
        tf.print(f"Derived 'window_size' from 3D preprocessed training data: {config['window_size']}")
    elif x_train_data.shape[1] != config['window_size']:
         tf.print(f"Warning: Mismatch between explicitly set config 'window_size' ({config['window_size']}) and "
               f"3D x_train_data.shape[1] ({x_train_data.shape[1]}). Using config 'window_size'.")

    config['num_features_input'] = x_train_data.shape[2]
    tf.print(f"[DataPrep] Automatically set 'num_features_input': {config['num_features_input']} from x_train_data.shape[2]")

    config['num_features_output'] = y_train_targets_6_features.shape[1]
    tf.print(f"[DataPrep] Automatically set 'num_features_output': {config['num_features_output']} from y_train_targets_6_features.shape[1]")

    required_dims = {
        'rnn_hidden_dim': "RNN hidden state dimension for CVAE components.",
        'conditioning_dim': "Conditioning vector dimension for CVAE components.",
        'latent_dim': "Latent space dimension for CVAE."
    }
    for dim_key, description in required_dims.items():
        if dim_key not in config or not isinstance(config.get(dim_key), int) or config.get(dim_key) <= 0:
            raise ValueError(f"'{dim_key}' not found in config or is not a positive integer. {description} It's required.")

    num_train_samples = x_train_data.shape[0]
    h_context_train = datasets.get("h_train")
    if h_context_train is None:
        tf.print(f"Warning: 'h_train' not found in datasets. Generating zeros for h_context_train (shape: ({num_train_samples}, {config['rnn_hidden_dim']})).")
        h_context_train = np.zeros((num_train_samples, config['rnn_hidden_dim']), dtype=np.float32)
    elif h_context_train.shape != (num_train_samples, config['rnn_hidden_dim']):
        raise ValueError(f"Shape mismatch for h_context_train. Expected ({num_train_samples}, {config['rnn_hidden_dim']}), got {h_context_train.shape}")

    conditions_t_train = datasets.get("cond_train")
    if conditions_t_train is None:
        tf.print(f"Warning: 'cond_train' not found in datasets. Generating zeros for conditions_t_train (shape: ({num_train_samples}, {config['conditioning_dim']})).")
        conditions_t_train = np.zeros((num_train_samples, config['conditioning_dim']), dtype=np.float32)
    elif conditions_t_train.shape != (num_train_samples, config['conditioning_dim']):
        raise ValueError(f"Shape mismatch for conditions_t_train. Expected ({num_train_samples}, {config['conditioning_dim']}), got {conditions_t_train.shape}")

    cvae_train_inputs = [x_train_data, h_context_train, conditions_t_train]
    cvae_train_targets = y_train_targets_6_features

    # Validation data setup
    if x_val_data is not None and y_val_targets_6_features is not None:
        num_val_samples = x_val_data.shape[0]
        h_context_val = datasets.get("h_val")
        if h_context_val is None:
            tf.print(f"Warning: 'h_val' not found in datasets. Generating zeros for h_context_val (shape: ({num_val_samples}, {config['rnn_hidden_dim']})).")
            h_context_val = np.zeros((num_val_samples, config['rnn_hidden_dim']), dtype=np.float32)
        elif h_context_val.shape != (num_val_samples, config['rnn_hidden_dim']):
            raise ValueError(f"Shape mismatch for h_context_val. Expected ({num_val_samples}, {config['rnn_hidden_dim']}), got {h_context_val.shape}")

        conditions_t_val = datasets.get("cond_val")
        if conditions_t_val is None:
            tf.print(f"Warning: 'cond_val' not found in datasets. Generating zeros for conditions_t_val (shape: ({num_val_samples}, {config['conditioning_dim']})).")
            conditions_t_val = np.zeros((num_val_samples, config['conditioning_dim']), dtype=np.float32)
        elif conditions_t_val.shape != (num_val_samples, config['conditioning_dim']):
            raise ValueError(f"Shape mismatch for conditions_t_val. Expected ({num_val_samples}, {config['conditioning_dim']}), got {conditions_t_val.shape}")

        config['cvae_val_inputs'] = {
            'x_window': x_val_data,
            'h_context': h_context_val,
            'conditions_t': conditions_t_val
        }
        config['cvae_val_targets'] = y_val_targets_6_features
        tf.print(f"[data_processor] Added cvae_val_inputs and cvae_val_targets to config from preprocessor output.")
    else:
        tf.print("[data_processor] Validation data (x_val_data or y_val_targets_6_features from preprocessor) is None. "
                 "Training will proceed without validation unless 'cvae_val_inputs' and 'cvae_val_targets' are already in config.")
        # If not provided by preprocessor, check if they were loaded from a config file
        if 'cvae_val_inputs' not in config or 'cvae_val_targets' not in config:
            config.pop('cvae_val_inputs', None) # Ensure they are removed if incomplete
            config.pop('cvae_val_targets', None)
            tf.print("[data_processor] No validation data found in preprocessor output or existing config.")


    initial_latent_dim = config.get('initial_latent_dim', config['latent_dim']) 
    step_size_latent = config.get('step_size_latent', 8) 
    threshold_error = config.get('threshold_error', 0.1) # MAE threshold
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 128)
    incremental_search = config.get('incremental_search', False) 

    current_latent_dim = initial_latent_dim
    best_val_mae = float('inf')
    best_latent_dim = current_latent_dim
    best_autoencoder_manager = None
    
    expected_mae_key = 'reconstruction_out_mae'
    expected_val_mae_key = f'val_{expected_mae_key}'

    while True:
        config['latent_dim'] = current_latent_dim 
        tf.print(f"Training CVAE with latent_dim: {current_latent_dim}")
        
        autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
        autoencoder_manager.build_autoencoder(config) 
        
        autoencoder_manager.train_autoencoder(
            data_inputs=cvae_train_inputs,
            data_targets=cvae_train_targets,
            epochs=epochs,
            batch_size=batch_size,
            config=config 
        )
        
        train_eval_results = autoencoder_manager.evaluate(cvae_train_inputs, cvae_train_targets, "Training", config)
        
        training_mae = float('nan')
        mae_metric_key_train_found = None
        if train_eval_results:
            if expected_mae_key in train_eval_results:
                training_mae = train_eval_results[expected_mae_key]
                mae_metric_key_train_found = expected_mae_key
            else: # Fallback search
                for key in train_eval_results.keys():
                    if 'mae' in key.lower() and 'reconstruction_out' in key.lower():
                        training_mae = train_eval_results[key]
                        mae_metric_key_train_found = key
                        tf.print(f"Warning: Training MAE key '{expected_mae_key}' not found. Using fallback key '{mae_metric_key_train_found}'.")
                        break
        
        tf.print(f"Training MAE with latent_dim {current_latent_dim}: {training_mae:.4f} (extracted with key: {mae_metric_key_train_found})")
        tf.print(f"Full Training Evaluation Results: {train_eval_results}")

        validation_mae = float('nan') # Default to NaN if no validation
        mae_metric_key_val_found = None
        if config.get('cvae_val_inputs') and config.get('cvae_val_targets') is not None:
            eval_val_inputs_list = [
                config['cvae_val_inputs']['x_window'],
                config['cvae_val_inputs']['h_context'],
                config['cvae_val_inputs']['conditions_t']
            ]
            eval_val_targets_array = config['cvae_val_targets']
            
            val_eval_results = autoencoder_manager.evaluate(eval_val_inputs_list, eval_val_targets_array, "Validation", config)
            
            if val_eval_results:
                if expected_val_mae_key in val_eval_results:
                    validation_mae = val_eval_results[expected_val_mae_key]
                    mae_metric_key_val_found = expected_val_mae_key
                else: # Fallback search
                    for key in val_eval_results.keys():
                        if 'mae' in key.lower() and 'reconstruction_out' in key.lower() and 'val_' in key.lower():
                            validation_mae = val_eval_results[key]
                            mae_metric_key_val_found = key
                            tf.print(f"Warning: Validation MAE key '{expected_val_mae_key}' not found. Using fallback key '{mae_metric_key_val_found}'.")
                            break
            tf.print(f"Validation MAE with latent_dim {current_latent_dim}: {validation_mae:.4f} (extracted with key: {mae_metric_key_val_found})")
            tf.print(f"Full Validation Evaluation Results: {val_eval_results}")
        else:
            tf.print(f"Validation data not configured for latent_dim {current_latent_dim}. Skipping validation MAE comparison for search.")
        
        if not np.isnan(validation_mae) and validation_mae < best_val_mae:
            best_val_mae = validation_mae
            best_latent_dim = current_latent_dim
            best_autoencoder_manager = autoencoder_manager 
            tf.print(f"New best Validation MAE: {best_val_mae:.4f} with latent_dim: {best_latent_dim}")
        elif np.isnan(validation_mae) and best_autoencoder_manager is None: # First run, no validation, save current
            best_latent_dim = current_latent_dim
            best_autoencoder_manager = autoencoder_manager
            tf.print(f"No validation MAE. Storing model for latent_dim: {current_latent_dim} as current best.")


        if not incremental_search:
            tf.print("Incremental search disabled. Completing after one iteration.")
            if best_autoencoder_manager is None: best_autoencoder_manager = autoencoder_manager # Ensure a model is selected
            break 
        
        if not np.isnan(validation_mae) and validation_mae <= threshold_error:
            tf.print(f"Threshold MAE ({threshold_error}) met or improved. Optimal latent_dim likely found: {current_latent_dim}")
            break
        else:
            current_latent_dim += step_size_latent
            if current_latent_dim > config.get('max_latent_dim', 256):
                tf.print(f"Reached max_latent_dim ({config.get('max_latent_dim', 256)}). Stopping search.")
                break
            if current_latent_dim <= 0: # Should not happen with positive step_size
                tf.print(f"Latent dimension became non-positive ({current_latent_dim}). Stopping search.")
                break
    
    if best_autoencoder_manager is None: 
        if autoencoder_manager: # If loop ran at least once
            tf.print("Warning: best_autoencoder_manager was not set (e.g. all val_mae were NaN or search conditions not met). Using the last trained model.")
            best_autoencoder_manager = autoencoder_manager
            best_latent_dim = config.get('latent_dim') # Get the last used latent_dim
        else: # Should not happen if pipeline starts correctly
            raise RuntimeError("Autoencoder training loop did not run or failed to select a model.")

    tf.print(f"Final selected latent_dim: {best_latent_dim} with Best Validation MAE: {best_val_mae:.4f}")
    
    config['latent_dim'] = best_latent_dim # Ensure config reflects the best latent_dim for final ops

    encoder_model_filename = config.get('save_encoder', 'encoder_model.keras').replace(".keras", f"_ld{best_latent_dim}.keras").replace(".h5", f"_ld{best_latent_dim}.keras")
    decoder_model_filename = config.get('save_decoder', 'decoder_model.keras').replace(".keras", f"_ld{best_latent_dim}.keras").replace(".h5", f"_ld{best_latent_dim}.keras")
    
    if best_autoencoder_manager.encoder_plugin:
        best_autoencoder_manager.save_encoder(encoder_model_filename) # Uses plugin's save
        tf.print(f"Saved best encoder component model to {encoder_model_filename}")
    if best_autoencoder_manager.decoder_plugin:
        best_autoencoder_manager.save_decoder(decoder_model_filename) # Uses plugin's save
        tf.print(f"Saved best decoder component model to {decoder_model_filename}")
    
    # Save the full CVAE model if a path is provided in config
    full_model_save_path_template = config.get('model_save_path', None)
    if full_model_save_path_template:
        full_model_filename = full_model_save_path_template.replace(".keras", f"_ld{best_latent_dim}.keras").replace(".h5", f"_ld{best_latent_dim}.keras")
        best_autoencoder_manager.save_model(full_model_filename) # Saves the full Keras model
        tf.print(f"Saved best full CVAE model to {full_model_filename}")


    end_time = time.time()
    execution_time = end_time - start_time
    
    final_train_eval_results = best_autoencoder_manager.evaluate(cvae_train_inputs, cvae_train_targets, "Final Training", config)
    final_training_mae = float('nan')
    final_mae_metric_key_train_found = None
    if final_train_eval_results:
        if expected_mae_key in final_train_eval_results:
            final_training_mae = final_train_eval_results[expected_mae_key]
            final_mae_metric_key_train_found = expected_mae_key
        else: # Fallback
            for key in final_train_eval_results.keys():
                if 'mae' in key.lower() and 'reconstruction_out' in key.lower():
                    final_training_mae = final_train_eval_results[key]
                    final_mae_metric_key_train_found = key
                    break
    tf.print(f"Final Training MAE (best model): {final_training_mae:.4f} (key: {final_mae_metric_key_train_found})")


    final_val_eval_results = None
    final_validation_mae = float('nan')
    final_mae_metric_key_val_found = None
    if config.get('cvae_val_inputs') and config.get('cvae_val_targets') is not None:
        final_eval_val_inputs_list = [
            config['cvae_val_inputs']['x_window'],
            config['cvae_val_inputs']['h_context'],
            config['cvae_val_inputs']['conditions_t']
        ]
        final_eval_val_targets_array = config['cvae_val_targets']
        final_val_eval_results = best_autoencoder_manager.evaluate(final_eval_val_inputs_list, final_eval_val_targets_array, "Final Validation", config)
        if final_val_eval_results:
            if expected_val_mae_key in final_val_eval_results:
                final_validation_mae = final_val_eval_results[expected_val_mae_key]
                final_mae_metric_key_val_found = expected_val_mae_key
            else: # Fallback
                for key in final_val_eval_results.keys():
                    if 'mae' in key.lower() and 'reconstruction_out' in key.lower() and 'val_' in key.lower():
                        final_validation_mae = final_val_eval_results[key]
                        final_mae_metric_key_val_found = key
                        break
    tf.print(f"Final Validation MAE (best model): {final_validation_mae:.4f} (key: {final_mae_metric_key_val_found})")


    debug_info = {
        'execution_time_seconds': execution_time,
        'best_latent_dim': best_latent_dim,
        'encoder_plugin_params': best_autoencoder_manager.encoder_plugin.get_debug_info() if best_autoencoder_manager.encoder_plugin else None,
        'decoder_plugin_params': best_autoencoder_manager.decoder_plugin.get_debug_info() if best_autoencoder_manager.decoder_plugin else None,
        'final_validation_mae': final_validation_mae if not np.isnan(final_validation_mae) else None,
        'final_training_mae': final_training_mae if not np.isnan(final_training_mae) else None,
        'final_validation_metrics': final_val_eval_results,
        'final_training_metrics': final_train_eval_results,
        'config_used': {k: v for k, v in config.items() if not isinstance(v, np.ndarray)} # Avoid saving large arrays in debug log
    }

    from tensorflow.keras.utils import plot_model
    if 'save_log' in config and config['save_log']:
        save_debug_info(debug_info, config['save_log'])
        tf.print(f"Debug info saved to {config['save_log']}.")
    
    if 'remote_log' in config and config['remote_log']:
        remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
        tf.print(f"Debug info saved to {config['remote_log']}.")
    
    model_plot_file_template = config.get('model_plot_file', None)
    if model_plot_file_template and best_autoencoder_manager.autoencoder_model:
        model_plot_filename = model_plot_file_template.replace(".png", f"_ld{best_latent_dim}.png")
        try: 
            plot_model(best_autoencoder_manager.autoencoder_model, to_file=model_plot_filename, show_shapes=True, show_layer_names=True, expand_nested=True)
            tf.print(f"Model plot saved to {model_plot_filename}")
        except Exception as e: 
            tf.print(f"Could not plot model: {e}")
        
    tf.print(f"Pipeline execution time: {execution_time:.2f} seconds")


def load_and_evaluate_encoder(config):
    custom_objects = {} 
    # encoder_plugin_name = config.get('encoder_plugin', '').lower()
    # if encoder_plugin_name == 'your_plugin_name_if_it_has_custom_layers':
    #     from app.plugins.your_encoder_plugin import CustomLayerForEncoder # Example
    #     custom_objects['CustomLayerForEncoder'] = CustomLayerForEncoder
    
    encoder_model_path = config['load_encoder']
    if not encoder_model_path or not os.path.exists(encoder_model_path):
        print(f"Encoder model path not found or not specified: {encoder_model_path}")
        return

    # This loads the Keras model of the encoder *component*
    encoder_component_model = load_model(encoder_model_path, custom_objects=custom_objects, compile=False)
    print(f"CVAE Encoder component model loaded from {encoder_model_path}")

    x_input_file = config.get('x_test_file', config.get('x_validation_file')) # Prioritize test, then val
    if not x_input_file: raise ValueError("No input file (x_test_file or x_validation_file) specified for encoder evaluation.")

    # Load raw input data (e.g., windowed data as produced by preprocessor)
    # This should be the X_window part of the CVAE input.
    # We assume the preprocessor has already windowed it.
    # The preprocessor should return 'x_test' or 'x_val' in 3D shape (samples, window_size, num_features_input)
    
    # For evaluation, we need to run the preprocessor on the test/val data specified
    # This is a simplified approach; ideally, preprocessor is run once and datasets are split.
    # Here, we re-run preprocessor for the specified file to get x_test_data.
    temp_eval_config = config.copy()
    # Temporarily override train/val files to point to the test file for preprocessing
    temp_eval_config['x_train_file'] = x_input_file 
    temp_eval_config['y_train_file'] = x_input_file # Assuming y_file is same for autoencoder preprocessing
    temp_eval_config['x_validation_file'] = None # Don't need validation split here
    temp_eval_config['y_validation_file'] = None
    
    # Instantiate a new preprocessor to avoid state issues if the main one is stateful
    # This requires preprocessor_plugin to be an importable class path or similar
    from app.plugin_loader import load_plugin # Assuming you have a plugin loader
    preprocessor_class = load_plugin(config['preprocessor_plugin'], 'preprocessor_plugins')
    eval_preprocessor = preprocessor_class()

    eval_datasets = eval_preprocessor.run_preprocessing(temp_eval_config)
    x_window_eval_data = eval_datasets.get("x_train") # Preprocessor output for 'x_train' is our eval data

    if x_window_eval_data is None or x_window_eval_data.ndim != 3:
        raise ValueError(f"Evaluation data 'x_window_eval_data' from preprocessor is not correctly shaped or is None. Shape: {getattr(x_window_eval_data, 'shape', 'N/A')}")

    num_samples_eval = x_window_eval_data.shape[0]

    rnn_hidden_dim_eval = config.get('rnn_hidden_dim') 
    conditioning_dim_eval = config.get('conditioning_dim')
    if rnn_hidden_dim_eval is None or conditioning_dim_eval is None:
        raise ValueError("rnn_hidden_dim and conditioning_dim must be in config for encoder evaluation.")

    # Generate h_context and conditions_t for evaluation
    # These could also come from eval_datasets if preprocessor provides them ('h_train', 'cond_train')
    h_context_eval = eval_datasets.get("h_train")
    if h_context_eval is None:
        print(f"Generating placeholder h_context for encoder evaluation (shape: ({num_samples_eval}, {rnn_hidden_dim_eval})).")
        h_context_eval = np.zeros((num_samples_eval, rnn_hidden_dim_eval), dtype=np.float32)
    elif h_context_eval.shape[0] != num_samples_eval or h_context_eval.shape[1] != rnn_hidden_dim_eval:
        raise ValueError(f"Shape mismatch for eval h_context. Expected ({num_samples_eval}, {rnn_hidden_dim_eval}), got {h_context_eval.shape}")


    conditions_t_eval = eval_datasets.get("cond_train")
    if conditions_t_eval is None:
        print(f"Generating placeholder conditions_t for encoder evaluation (shape: ({num_samples_eval}, {conditioning_dim_eval})).")
        conditions_t_eval = np.zeros((num_samples_eval, conditioning_dim_eval), dtype=np.float32)
    elif conditions_t_eval.shape[0] != num_samples_eval or conditions_t_eval.shape[1] != conditioning_dim_eval:
        raise ValueError(f"Shape mismatch for eval conditions_t. Expected ({num_samples_eval}, {conditioning_dim_eval}), got {conditions_t_eval.shape}")
    
    encoder_inputs_eval = [x_window_eval_data, h_context_eval, conditions_t_eval]

    print(f"Encoding data. Input shapes: x_window: {encoder_inputs_eval[0].shape}, h_context: {encoder_inputs_eval[1].shape}, conditions_t: {encoder_inputs_eval[2].shape}")
    encoded_outputs = encoder_component_model.predict(encoder_inputs_eval, verbose=1)
    z_mean_data = encoded_outputs[0]
    # z_log_var_data = encoded_outputs[1] 
    
    print(f"Encoded z_mean shape: {z_mean_data.shape}")

    if config.get('evaluate_encoder'):
        output_filename = config['evaluate_encoder']
        encoded_df = pd.DataFrame(z_mean_data)
        encoded_df.columns = [f'latent_feature_{i}' for i in range(z_mean_data.shape[1])]
        
        # Date handling: If the preprocessor output 'x_train_dates' (or similar) exists and aligns
        eval_dates = eval_datasets.get("x_train_dates") # Assuming preprocessor might return dates
        if eval_dates is not None and len(eval_dates) == len(encoded_df):
            encoded_df.index = pd.to_datetime(eval_dates)
            print(f"Added dates to encoded output from preprocessor.")
        
        write_csv(encoded_df, output_filename, index=eval_dates is not None) # Write index if dates were added
        print(f"Encoded z_mean data saved to {output_filename}")


def load_and_evaluate_decoder(config):
    decoder_model_path = config['load_decoder']
    if not decoder_model_path or not os.path.exists(decoder_model_path):
        print(f"Decoder model path not found or not specified: {decoder_model_path}")
        return
        
    decoder_component_model = load_model(decoder_model_path, compile=False)
    print(f"CVAE Decoder component model loaded from {decoder_model_path}")

    # Input for decoder evaluation should be z_t samples.
    # This could be the output of the encoder evaluation, or a separate file of z_t samples.
    z_t_input_file = config.get('evaluate_encoder_output_for_decoder', config.get('evaluate_encoder')) # Use encoder output if specified
    if not z_t_input_file:
        # Fallback to a generic input file if evaluate_encoder is not set for this purpose
        z_t_input_file = config.get('x_test_file_for_z_samples') # A new config key might be needed
        if not z_t_input_file:
            raise ValueError("No input file specified for z_t samples for decoder evaluation.")
    
    if not os.path.exists(z_t_input_file):
        raise FileNotFoundError(f"Input file for z_t samples not found: {z_t_input_file}")

    # Load z_t samples. Assume it's a CSV without dates unless 'force_date' and headers suggest otherwise.
    z_t_df = load_csv(file_path=z_t_input_file, headers=True, force_date=False) # Assume headers like latent_feature_0
    z_t_data = z_t_df.to_numpy()

    num_samples_eval = z_t_data.shape[0]
    if z_t_data.shape[1] != config.get('latent_dim'):
         print(f"Warning: Loaded z_t data feature count ({z_t_data.shape[1]}) does not match config's latent_dim ({config.get('latent_dim')}). Ensure correct z_t input file.")
    
    rnn_hidden_dim_eval = config.get('rnn_hidden_dim')
    conditioning_dim_eval = config.get('conditioning_dim')
    if rnn_hidden_dim_eval is None or conditioning_dim_eval is None:
        raise ValueError("rnn_hidden_dim and conditioning_dim must be in config for decoder evaluation.")

    # Generate h_context and conditions_t for evaluation
    # These might need to be loaded if they correspond to the z_t samples
    # For simplicity, generating zeros.
    print(f"Generating placeholder h_context and conditions_t for decoder evaluation.")
    h_context_eval = np.zeros((num_samples_eval, rnn_hidden_dim_eval), dtype=np.float32)
    conditions_t_eval = np.zeros((num_samples_eval, conditioning_dim_eval), dtype=np.float32)

    decoder_inputs_eval = [z_t_data, h_context_eval, conditions_t_eval]

    print(f"Decoding data. Input shapes: z_t: {decoder_inputs_eval[0].shape}, h_context: {decoder_inputs_eval[1].shape}, conditions_t: {decoder_inputs_eval[2].shape}")
    decoded_data = decoder_component_model.predict(decoder_inputs_eval, verbose=1)
    print(f"Decoded data (reconstructed x_prime_t) shape: {decoded_data.shape}")

    if config.get('evaluate_decoder'):
        output_filename = config['evaluate_decoder']
        decoded_df = pd.DataFrame(decoded_data)
        
        # Use the target feature names for columns
        target_feature_names_eval = config.get('cvae_target_feature_names', ['OPEN', 'LOW', 'HIGH', 'vix_close', 'BC-BO', 'BH-BL'])
        if len(target_feature_names_eval) == decoded_data.shape[1]:
            decoded_df.columns = target_feature_names_eval
        else:
            decoded_df.columns = [f'reconstructed_feature_{i}' for i in range(decoded_data.shape[1])]
            print(f"Warning: Number of target feature names ({len(target_feature_names_eval)}) "
                  f"does not match decoded data features ({decoded_data.shape[1]}). Using generic column names.")

        # Date handling: If the input z_t_df had an index (e.g., dates), try to use it.
        if isinstance(z_t_df.index, pd.DatetimeIndex):
            decoded_df.index = z_t_df.index
            print(f"Added dates to decoded output from z_t input file.")
        
        write_csv(decoded_df, output_filename, index=isinstance(z_t_df.index, pd.DatetimeIndex))
        print(f"Decoded data saved to {output_filename}")

# --- Remove the redundant num_features_input/output setup block at the end ---
# The logic for setting these is now at the beginning of run_autoencoder_pipeline.
# The block starting with:
# # In data_processor.py (inside run_autoencoder_pipeline) or main.py before calling it
# ...
# if x_window_train is not None and x_window_train.ndim == 3:
# down to 
#     raise ValueError("y_train_recon is not correctly shaped or is None. Cannot determine num_features_output.")
# should be deleted.



