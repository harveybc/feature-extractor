import numpy as np
import pandas as pd # Added for DataFrame creation in evaluate functions
import time # Added for execution_time
import tensorflow as tf # Added for tf.print consistency
from app.autoencoder_manager import AutoencoderManager
from app.data_handler import load_csv, write_csv
# from app.reconstruction import unwindow_data 
from app.config_handler import save_debug_info, remote_log, sanitize_dict_for_json
import os # Add os import for load_and_evaluate_encoder/decoder path checks
from tensorflow.keras.models import load_model # Changed
from tensorflow.keras.utils import plot_model # Changed
import matplotlib.pyplot as plt
import traceback # Ensure traceback is imported for detailed error printing


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


def calculate_datetime_features(timestamps):
    """
    Calculate the 10 cyclical/seasonal features for each timestamp.
    Args:
        timestamps (pd.Series or DatetimeIndex): Timestamps for each sample.
    Returns:
        np.ndarray: (num_samples, 10) array of cyclical features.
    """
    hours = timestamps.hour
    days_of_week = timestamps.dayofweek
    days_of_month = timestamps.day
    months = timestamps.month
    days_of_year = timestamps.dayofyear
    return np.stack([
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * days_of_week / 7),
        np.cos(2 * np.pi * days_of_week / 7),
        np.sin(2 * np.pi * days_of_month / 31),
        np.cos(2 * np.pi * days_of_month / 31),
        np.sin(2 * np.pi * months / 12),
        np.cos(2 * np.pi * months / 12),
        np.sin(2 * np.pi * days_of_year / 366),
        np.cos(2 * np.pi * days_of_year / 366),
    ], axis=1)


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
    tf.print(f"Number of CVAE target features: {len(target_feature_names)}") # ADDED: Print count
    
    try:
        target_indices = [feature_names_all.index(name) for name in target_feature_names]
    except ValueError as e:
        missing_feature = str(e).split("'")[1] 
        raise ValueError(
            f"One of the CVAE target features ('{missing_feature}') not found in 'feature_names' "
            f"provided by PreprocessorPlugin. Available features: {feature_names_all}"
        ) from e

    if x_train_data.ndim != 3: # Expect 3D (samples, window_size, features)
        raise ValueError(f"x_train_data is expected to be 3D (samples, window, features), but got shape {x_train_data.shape}")
    
    y_train_targets = x_train_data[:, -1, target_indices] # RENAMED: from y_train_targets_6_features
    tf.print(f"Constructed y_train_targets with shape: {y_train_targets.shape}") # MODIFIED: Print statement

    y_val_targets = None # RENAMED: from y_val_targets_6_features
    if x_val_data is not None:
        if x_val_data.ndim != 3: # Expect 3D
            raise ValueError(f"x_val_data is expected to be 3D (samples, window, features), but got shape {x_val_data.shape}")
        y_val_targets = x_val_data[:, -1, target_indices] # RENAMED & Extract last step
        tf.print(f"Constructed y_val_targets with shape: {y_val_targets.shape}") # MODIFIED: Print statement
        if x_val_data.shape[0] != y_val_targets.shape[0]: # MODIFIED: Variable name
            raise ValueError(f"Sample count mismatch between x_val_data ({x_val_data.shape[0]}) and y_val_targets ({y_val_targets.shape[0]})") # MODIFIED
    else:
        tf.print("No x_val_data provided by preprocessor. Validation will be skipped if not set in config later.")


    if not isinstance(y_train_targets, np.ndarray): # MODIFIED: Variable name
        raise TypeError(f"y_train_targets is type {type(y_train_targets)}, expected np.ndarray.") # MODIFIED
    if y_val_targets is not None and not isinstance(y_val_targets, np.ndarray): # MODIFIED: Variable name
        raise TypeError(f"y_val_targets is type {type(y_val_targets)}, expected np.ndarray.") # MODIFIED


    # y_targets should be 2D (samples, num_target_features)
    if y_train_targets.ndim != 2 or y_train_targets.shape[-1] != len(target_feature_names):
        raise ValueError(f"y_train_targets should be 2D with {len(target_feature_names)} features, but got shape {y_train_targets.shape}")
    if y_val_targets is not None and (y_val_targets.ndim != 2 or y_val_targets.shape[-1] != len(target_feature_names)):
        raise ValueError(f"y_val_targets should be 2D with {len(target_feature_names)} features, but got shape {y_val_targets.shape}")
    

    if x_train_data.shape[0] != y_train_targets.shape[0]: # MODIFIED: Variable name
        raise ValueError(f"Sample count mismatch between x_train_data ({x_train_data.shape[0]}) and y_train_targets ({y_train_targets.shape[0]})") # MODIFIED

    # --- Populate necessary dimensions in config ---
    if 'window_size' not in config or not isinstance(config.get('window_size'), int) or config.get('window_size') <= 0:
        if x_train_data.ndim == 3: # Check if x_train_data is 3D
            config['window_size'] = x_train_data.shape[1]
            tf.print(f"Derived 'window_size' from 3D preprocessed training data: {config['window_size']}")
        else: # Handle case where x_train_data is not 3D and window_size is not in config
            raise ValueError("x_train_data is not 3D and 'window_size' is not specified in config. Cannot derive window_size.")
    elif x_train_data.ndim == 3 and x_train_data.shape[1] != config['window_size']: # If 3D and mismatch
         tf.print(f"Warning: Mismatch between explicitly set config 'window_size' ({config['window_size']}) and "
               f"3D x_train_data.shape[1] ({x_train_data.shape[1]}). Using config 'window_size'.")
    elif x_train_data.ndim != 3: # If not 3D but window_size is in config
        tf.print(f"Warning: 'window_size' is set in config to {config['window_size']}, but x_train_data is not 3D (shape: {x_train_data.shape}). Proceeding with config 'window_size'. Ensure model handles this.")


    if x_train_data.ndim == 3:
        config['num_features_input'] = x_train_data.shape[2]
    elif x_train_data.ndim == 2: # If preprocessor returns 2D (samples, features) and windowing is internal to model
        config['num_features_input'] = x_train_data.shape[1]
        tf.print(f"Warning: x_train_data is 2D. Assuming 'num_features_input' is x_train_data.shape[1]. Ensure model handles windowing if needed.")
    else: # Should have been caught by earlier x_train_data.ndim != 3 check
        raise ValueError(f"Cannot determine 'num_features_input'. x_train_data has unexpected shape: {x_train_data.shape}")
    tf.print(f"[DataPrep] Automatically set 'num_features_input': {config['num_features_input']}")

    config['num_features_output'] = y_train_targets.shape[1] # MODIFIED: num_features_output is from the target
    tf.print(f"[DataPrep] Automatically set 'num_features_output': {config['num_features_output']} from y_train_targets.shape[1]") # MODIFIED

    required_dims = {
        'rnn_hidden_dim': "RNN hidden state dimension for CVAE components.",
        'conditioning_dim': "Conditioning vector dimension for CVAE components.",
        'latent_dim': "Latent space dimension for CVAE."
    }
    for dim_key, description in required_dims.items():
        if dim_key not in config or not isinstance(config.get(dim_key), int) or config.get(dim_key) <= 0:
            raise ValueError(f"'{dim_key}' not found in config or is not a positive integer. {description} It's required.")

    num_train_samples = x_train_data.shape[0]
    # --- PATCH: Build h_context_train as zeros of shape (num_samples, rnn_hidden_dim) ---
    h_context_train = np.zeros((num_train_samples, config['rnn_hidden_dim']), dtype=np.float32)
    # This matches the model's context input expectation and does not touch any other data or features.

    # --- PATCH: Build conditions_t_train from timestamps ---
    # Try to get timestamps from datasets, else from config
    timestamps = None
    if 'x_train_dates' in datasets:
        timestamps = pd.to_datetime(datasets['x_train_dates'])
    elif 'timestamps' in datasets:
        timestamps = pd.to_datetime(datasets['timestamps'])
    elif 'timestamps' in config:
        timestamps = pd.to_datetime(config['timestamps'])
    if timestamps is None or len(timestamps) != num_train_samples:
        raise ValueError("Timestamps required to build conditions_t_train were not found or length mismatch.")
    conditions_t_train = calculate_datetime_features(timestamps)

    cvae_train_inputs = [x_train_data, h_context_train, conditions_t_train]
    cvae_train_targets = y_train_targets # MODIFIED: Variable name

    # Validation data setup
    if x_val_data is not None and y_val_targets is not None: # MODIFIED: Variable name
        num_val_samples = x_val_data.shape[0]
        h_context_val = datasets.get("h_val")
        if h_context_val is None:
            tf.print(f"Warning: 'h_val' not found in datasets. Generating zeros for h_context_val (shape: ({num_val_samples}, {config['rnn_hidden_dim']})).")
            h_context_val = np.zeros((num_val_samples, config['rnn_hidden_dim']), dtype=np.float32)
        elif h_context_val.shape != (num_val_samples, config['rnn_hidden_dim']):
            raise ValueError(f"Shape mismatch for h_context_val. Expected ({num_val_samples}, {config['rnn_hidden_dim']}), got {h_context_val.shape}")

        # --- PATCH: Always compute conditions_t_val from validation timestamps if available ---
        val_timestamps = None
        if 'x_val_dates' in datasets:
            val_timestamps = pd.to_datetime(datasets['x_val_dates'])
        elif 'val_timestamps' in datasets:
            val_timestamps = pd.to_datetime(datasets['val_timestamps'])
        elif 'val_timestamps' in config:
            val_timestamps = pd.to_datetime(config['val_timestamps'])
        if val_timestamps is not None and len(val_timestamps) == num_val_samples:
            conditions_t_val = calculate_datetime_features(val_timestamps)
        else:
            raise ValueError("Timestamps required to build conditions_t_val for validation were not found or length mismatch.")

        config['cvae_val_inputs'] = {
            'x_window': x_val_data,
            'h_context': h_context_val,
            'conditions_t': conditions_t_val
        }
        config['cvae_val_targets'] = y_val_targets
        tf.print(f"[data_processor] Added cvae_val_inputs and cvae_val_targets to config from preprocessor output.")
    else:
        tf.print("[data_processor] Validation data (x_val_data or y_val_targets from preprocessor) is None. "
                 "Training will proceed without validation unless 'cvae_val_inputs' and 'cvae_val_targets' are already in config.")
        if 'cvae_val_inputs' not in config or 'cvae_val_targets' not in config:
            config.pop('cvae_val_inputs', None)
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
    best_history = None # ADDED: To store history of the best model
    
    # This key should now be 'reconstruction_out_custom_recon_mae' 
    # because autoencoder_manager.py is using MeanAbsoluteError(name='custom_recon_mae').
    # Keras prepends the output name 'reconstruction_out'.
    # If 'mae' (string) is used in compile, key is 'reconstruction_out_mae'.
    # If tf.keras.metrics.MeanAbsoluteError() instance is used (without name), key is 'reconstruction_out_mean_absolute_error'.
    # NEW: With dedicated output 'reconstruction_out_for_mae_calc' and MeanAbsoluteError metric:
    expected_mae_key = 'reconstruction_out_for_mae_calc_mean_absolute_error' # UPDATED EXPECTED KEY
    expected_val_mae_key = f'val_{expected_mae_key}' # Keras prepends 'val_'

    while True:
        config['latent_dim'] = current_latent_dim 
        tf.print(f"Training CVAE with latent_dim: {current_latent_dim}")
        
        autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
        autoencoder_manager.build_autoencoder(config) 
        
        history = autoencoder_manager.train_autoencoder( # MODIFIED: Capture history
            data_inputs=cvae_train_inputs,
            data_targets=cvae_train_targets,
            epochs=epochs,
            batch_size=batch_size,
            config=config 
        )
        
        train_eval_results = autoencoder_manager.evaluate(cvae_train_inputs, cvae_train_targets, "Training", config)
        
        training_mae = float('nan')
        if train_eval_results and expected_mae_key in train_eval_results:
            training_mae = train_eval_results[expected_mae_key]
        
        if np.isnan(training_mae):
            tf.print(f"CRITICAL ERROR: Training MAE ('{expected_mae_key}') is NaN or key not found in evaluation results!")
            tf.print(f"Evaluation results for Training (train_eval_results):")
            if isinstance(train_eval_results, dict):
                for k_iter, v_iter in train_eval_results.items():
                    tf.print(f"  Key: '{k_iter}', Value: {v_iter}, Type: {type(v_iter)}")
            else:
                tf.print(f"  train_eval_results is not a dict: {train_eval_results}")
            
            compiled_metrics_names_str = "N/A"
            if hasattr(autoencoder_manager, 'autoencoder_model') and autoencoder_manager.autoencoder_model and hasattr(autoencoder_manager.autoencoder_model, 'metrics_names'):
                 compiled_metrics_names_str = str(autoencoder_manager.autoencoder_model.metrics_names)
            tf.print(f"Model's compiled metrics_names attribute: {compiled_metrics_names_str}")
            tf.print(f"Model's actual metric objects (name and class):")
            if hasattr(autoencoder_manager, 'autoencoder_model') and autoencoder_manager.autoencoder_model and hasattr(autoencoder_manager.autoencoder_model, 'metrics'):
                for m_obj in autoencoder_manager.autoencoder_model.metrics:
                    tf.print(f"  - {m_obj.name} ({m_obj.__class__.__name__})")
            else:
                tf.print("  Could not retrieve model.metrics objects.")
            raise ValueError(f"Training MAE ('{expected_mae_key}') is missing or NaN. Stopping execution. Results: {train_eval_results}. Compiled metrics: {compiled_metrics_names_str}")
        
        tf.print(f"Training MAE with latent_dim {current_latent_dim}: {training_mae:.4f} (extracted with key: {expected_mae_key})")
        tf.print(f"Full Training Evaluation Results: {train_eval_results}")

        validation_mae = float('nan') 
        if config.get('cvae_val_inputs') and config.get('cvae_val_targets') is not None:
            eval_val_inputs_list = [
                config['cvae_val_inputs']['x_window'],
                config['cvae_val_inputs']['h_context'],
                config['cvae_val_inputs']['conditions_t']
            ]
            eval_val_targets_array = config['cvae_val_targets']
            
            val_eval_results = autoencoder_manager.evaluate(eval_val_inputs_list, eval_val_targets_array, "Validation", config)
            
            # Use expected_mae_key directly, as model.evaluate() does not add 'val_' prefix to keys in its returned dict
            if val_eval_results and expected_mae_key in val_eval_results: # CHANGED: Use expected_mae_key
                validation_mae = val_eval_results[expected_mae_key]       # CHANGED: Use expected_mae_key

            if np.isnan(validation_mae):
                # The expected_val_mae_key is what we *would* look for if Keras prefixed it, 
                # but for the error message, it's more informative to state what was attempted.
                # However, the actual lookup should use expected_mae_key.
                # For clarity in the error message if it still fails for other reasons, we can keep expected_val_mae_key here.
                tf.print(f"CRITICAL ERROR: Validation MAE (expected key pattern: '{expected_val_mae_key}', actual lookup key: '{expected_mae_key}') is NaN or key not found in evaluation results!") # MODIFIED ERROR MSG
                tf.print(f"Evaluation results for Validation (val_eval_results):")
                if isinstance(val_eval_results, dict):
                    for k_iter, v_iter in val_eval_results.items():
                        tf.print(f"  Key: '{k_iter}', Value: {v_iter}, Type: {type(v_iter)}") # Added print for value and type
                else:
                    tf.print(f"  val_eval_results is not a dict: {val_eval_results}")

                compiled_metrics_names_str = "N/A"
                if hasattr(autoencoder_manager, 'autoencoder_model') and autoencoder_manager.autoencoder_model and hasattr(autoencoder_manager.autoencoder_model, 'metrics_names'):
                    compiled_metrics_names_str = str(autoencoder_manager.autoencoder_model.metrics_names)
                tf.print(f"Model's compiled metrics_names attribute: {compiled_metrics_names_str}")
                tf.print(f"Model's actual metric objects (name and class):")
                if hasattr(autoencoder_manager, 'autoencoder_model') and autoencoder_manager.autoencoder_model and hasattr(autoencoder_manager.autoencoder_model, 'metrics'):
                    for m_obj in autoencoder_manager.autoencoder_model.metrics:
                        tf.print(f"  - {m_obj.name} ({m_obj.__class__.__name__})") # Added print for metric object
                else:
                    tf.print("  Could not retrieve model.metrics objects.")
                # The error message should reflect the key that was *intended* to be found for validation.
                raise ValueError(f"Validation MAE ('{expected_val_mae_key}' pattern, actual lookup with '{expected_mae_key}') is missing or NaN. Stopping execution. Results: {val_eval_results}. Compiled metrics: {compiled_metrics_names_str}") # MODIFIED ERROR MSG

            tf.print(f"Validation MAE with latent_dim {current_latent_dim}: {validation_mae:.4f} (extracted with key: {expected_mae_key})") # CHANGED: Use expected_mae_key
            tf.print(f"Full Validation Evaluation Results: {val_eval_results}")
        else:
            tf.print(f"Validation data not configured for latent_dim {current_latent_dim}. Skipping validation MAE comparison for search.")
        
        if not np.isnan(validation_mae) and validation_mae < best_val_mae:
            best_val_mae = validation_mae
            best_latent_dim = current_latent_dim
            best_autoencoder_manager = autoencoder_manager 
            best_history = history # ADDED: Store history of best model
            tf.print(f"New best Validation MAE: {best_val_mae:.4f} with latent_dim: {best_latent_dim}")
        elif np.isnan(validation_mae) and best_autoencoder_manager is None: 
            best_latent_dim = current_latent_dim
            best_autoencoder_manager = autoencoder_manager
            best_history = history # ADDED: Store history if no validation but it's the first/only model
            tf.print(f"No validation MAE. Storing model for latent_dim: {current_latent_dim} as current best.")


        if not incremental_search:
            tf.print("Incremental search disabled. Completing after one iteration.")
            if best_autoencoder_manager is None: 
                best_autoencoder_manager = autoencoder_manager 
                best_history = history # ADDED: Ensure history is captured
            break 
        
        if not np.isnan(validation_mae) and validation_mae <= threshold_error:
            tf.print(f"Threshold MAE ({threshold_error}) met or improved. Optimal latent_dim likely found: {current_latent_dim}")
            break
        else: # Check if validation_mae is NaN before deciding to increment latent_dim
            if np.isnan(validation_mae) and not config.get('allow_search_without_val_mae', False): # Add a config flag if you want to allow search without val_mae
                tf.print(f"Validation MAE is NaN and search without val_mae is not allowed. Stopping search at latent_dim {current_latent_dim}.")
                if best_autoencoder_manager is None: 
                    best_autoencoder_manager = autoencoder_manager # Ensure a model is selected
                    best_history = history # ADDED: Ensure history is captured
                break

            current_latent_dim += step_size_latent
            if current_latent_dim > config.get('max_latent_dim', 256):
                tf.print(f"Reached max_latent_dim ({config.get('max_latent_dim', 256)}). Stopping search.")
                break
            if current_latent_dim <= 0: 
                tf.print(f"Latent dimension became non-positive ({current_latent_dim}). Stopping search.")
                break
    
    if best_autoencoder_manager is None: 
        if autoencoder_manager: # if loop ran at least once
            tf.print("Warning: best_autoencoder_manager was not set (e.g. incremental search off, or no validation improvement). Using the last trained model.")
            best_autoencoder_manager = autoencoder_manager
            # best_latent_dim is already config['latent_dim'] from the last iteration
            # best_history is already 'history' from the last iteration
            if best_history is None: # Should be set if autoencoder_manager exists
                 best_history = history # Ensure it's set from the very last 'history'
            if np.isnan(best_val_mae): # If no validation, best_val_mae might be inf
                best_val_mae = validation_mae # Log the last validation_mae (even if NaN)
        else: 
            raise RuntimeError("Autoencoder training loop did not run or failed to select/produce a model.")


    best_val_mae_str = f"{best_val_mae:.4f}" if best_val_mae is not None and not np.isnan(best_val_mae) else "N/A"
    tf.print(f"Final selected latent_dim: {best_latent_dim} with Best/Final Validation MAE: {best_val_mae_str}")
    
    # config['latent_dim'] should already be best_latent_dim if search happened,
    # or the current_latent_dim from the last run.
    # For clarity, explicitly set it to best_latent_dim for saving.
    config['best_latent_dim_selected'] = best_latent_dim # Add this for clarity in saved config
    # The 'latent_dim' in config will be the one used for the selected model.

    # CRITICAL FIX: Save the trained encoder and decoder models
    if config.get('save_encoder') and best_autoencoder_manager:
        try:
            # Ensure directory exists
            encoder_save_path = config['save_encoder']
            encoder_dir = os.path.dirname(encoder_save_path)
            if encoder_dir:
                os.makedirs(encoder_dir, exist_ok=True)
            
            best_autoencoder_manager.save_encoder(encoder_save_path)
            tf.print(f"✅ Trained encoder saved to: {encoder_save_path}")
        except Exception as e:
            tf.print(f"❌ Failed to save encoder: {e}")
            tf.print(traceback.format_exc())
    elif config.get('save_encoder'):
        tf.print(f"⚠️  save_encoder specified ({config['save_encoder']}) but no trained model available")

    if config.get('save_decoder') and best_autoencoder_manager:
        try:
            # Ensure directory exists
            decoder_save_path = config['save_decoder']
            decoder_dir = os.path.dirname(decoder_save_path)
            if decoder_dir:
                os.makedirs(decoder_dir, exist_ok=True)
            
            best_autoencoder_manager.save_decoder(decoder_save_path)
            tf.print(f"✅ Trained decoder saved to: {decoder_save_path}")
        except Exception as e:
            tf.print(f"❌ Failed to save decoder: {e}")
            tf.print(traceback.format_exc())
    elif config.get('save_decoder'):
        tf.print(f"⚠️  save_decoder specified ({config['save_decoder']}) but no trained model available")

    # --- Plotting Training History ---
    loss_plot_file_template = config.get('loss_plot_file', None)
    if loss_plot_file_template and best_history is not None:
        # Ensure the directory exists
        plot_dir = os.path.dirname(loss_plot_file_template)
        if plot_dir: # If path includes a directory
            os.makedirs(plot_dir, exist_ok=True)
            
        loss_plot_filename = loss_plot_file_template.replace(".png", f"_ld{best_latent_dim}.png")
        tf.print(f"DEBUG: Attempting to plot. best_history.history keys: {list(best_history.history.keys())}")
        try:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            if 'loss' in best_history.history:
                plt.plot(best_history.history['loss'], label='Training Loss')
            else:
                tf.print("Warning: 'loss' key not found in best_history for plotting.")
            if 'val_loss' in best_history.history:
                plt.plot(best_history.history['val_loss'], label='Validation Loss')
            elif config.get('cvae_val_inputs'): # If validation was attempted
                tf.print("Warning: 'val_loss' key not found in best_history for plotting, but validation data was present.")
            plt.title(f'Model Loss (Latent Dim: {best_latent_dim})')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            
            mae_metric_key = 'reconstruction_out_for_mae_calc_mean_absolute_error'
            val_mae_metric_key = f'val_{mae_metric_key}'

            plt.subplot(2, 1, 2)
            if mae_metric_key in best_history.history:
                plt.plot(best_history.history[mae_metric_key], label='Training MAE')
            else:
                tf.print(f"Warning: Training MAE key '{mae_metric_key}' not found in best_history for plotting.")
            if val_mae_metric_key in best_history.history:
                plt.plot(best_history.history[val_mae_metric_key], label='Validation MAE')
            elif config.get('cvae_val_inputs'): # If validation was attempted
                tf.print(f"Warning: Validation MAE key '{val_mae_metric_key}' not found in best_history for plotting, but validation data was present.")
            plt.title(f'Model MAE (Latent Dim: {best_latent_dim})')
            plt.ylabel('Mean Absolute Error')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig(loss_plot_filename)
            plt.close() 
            tf.print(f"Training history plot saved to {loss_plot_filename}")
        except Exception as e:
            tf.print(f"Could not plot training history: {e}")
            tf.print(traceback.format_exc())
    elif loss_plot_file_template and best_history is None:
        tf.print(f"Loss plot file specified ({loss_plot_file_template}), but no training history available to plot.")

    end_time = time.time()
    execution_time = end_time - start_time

    # --- Prepare for debug_info ---
    # These _filter functions are okay for their specific purpose (extracting scalars from dicts)
    def _filter_metrics_dict_for_log(metrics_dict):
        # ... (your existing _filter_metrics_dict_for_log is likely fine, ensure it handles None robustly)
        if not isinstance(metrics_dict, dict): return {"error": "Metrics data was not a dictionary."}
        filtered = {}
        for k, v_metric in metrics_dict.items():
            if isinstance(v_metric, (int, float, np.integer, np.floating, str)):
                try:
                    filtered[k] = v_metric.item() if hasattr(v_metric, 'item') else v_metric
                except Exception as e_filter:
                    filtered[k] = f"Error converting metric '{k}': {str(e_filter)}"
            else:
                filtered[k] = f"<metric '{k}' type:{type(v_metric).__name__} removed>"
        return filtered

    def _filter_plugin_params_for_log(params):
        # ... (your existing _filter_plugin_params_for_log is likely fine)
        if not isinstance(params, dict): return "Plugin params not available or not a dict."
        return sanitize_dict_for_json(params) # Use the main sanitizer for consistency


    # For the config part of debug_info, select only essential keys.
    # The values will be taken from the 'config' dict which might still hold some complex objects
    # if not careful, but sanitize_dict_for_json applied to the whole debug_info later will catch them.
    # Or, build config_for_log_minimal by taking values and immediately sanitizing them.
    
    keys_for_debug_config = [
        'learning_rate', 'batch_size', 'epochs', 'window_size', 
        'num_features_input', 'num_features_output', 'rnn_hidden_dim', 
        'conditioning_dim', 'latent_dim', 'best_latent_dim_selected', # Use the one we added
        'kl_beta', 'kl_anneal_epochs', 'loss_function', 
        'reconstruction_loss_type', 'cvae_target_feature_names',
        'preprocessor_plugin', 'encoder_plugin', 'decoder_plugin',
        'x_train_file', 'x_validation_file', # Filenames are fine
        'save_encoder', 'save_decoder', 'model_save_path', 'loss_plot_file'
    ]
    config_subset_for_debug = {k: config.get(k) for k in keys_for_debug_config if k in config}
    # This config_subset_for_debug will be part of debug_info, which then gets sanitized.

    # Use MAE values from the best/last model run
    # 'training_mae' and 'validation_mae' should be from the scope of the selected best_autoencoder_manager
    # If incremental search was off, these are from the last run.
    # If incremental search was on, these should correspond to the 'best_val_mae' iteration.
    # This logic needs to ensure 'training_mae' and 'validation_mae' reflect the chosen model.
    
    # Re-evaluate the best model to get its specific training_mae if necessary,
    # or ensure 'training_mae' and 'validation_mae' from the loop correspond to 'best_autoencoder_manager'
    final_logged_training_mae = float('nan')
    final_logged_validation_mae = best_val_mae # This is already the best/final validation MAE

    # It's safer to re-evaluate the best model for its training MAE if incremental search was on
    # and best_autoencoder_manager is not the last 'autoencoder_manager' instance.
    # For simplicity now, we'll assume 'training_mae' from the loop's last relevant iteration is sufficient.
    # If 'best_autoencoder_manager' was chosen from an earlier iteration, 'training_mae' variable
    # from the end of the loop might not correspond to it.
    # Let's use the 'train_eval_results' and 'val_eval_results' that correspond to the 'best_history'.
    
    # If best_history exists, its last epoch contains the relevant metrics for that run.
    logged_train_metrics_from_history = {}
    logged_val_metrics_from_history = {}

    if best_history:
        for key_hist in best_history.history.keys():
            if not key_hist.startswith('val_'):
                logged_train_metrics_from_history[key_hist] = best_history.history[key_hist][-1] # Last epoch value
            else:
                logged_val_metrics_from_history[key_hist] = best_history.history[key_hist][-1] # Last epoch value
    
    # Extract MAE from these history-based metrics if available
    mae_key_from_history = 'reconstruction_out_for_mae_calc_mean_absolute_error'
    val_mae_key_from_history = f'val_{mae_key_from_history}'

    final_logged_training_mae = logged_train_metrics_from_history.get(mae_key_from_history, float('nan'))
    # final_logged_validation_mae is already best_val_mae

    debug_info = {
        'execution_time_seconds': execution_time,
        'best_latent_dim_selected': best_latent_dim,
        'final_training_mae_logged': final_logged_training_mae if not np.isnan(final_logged_training_mae) else None,
        'final_validation_mae_logged': final_logged_validation_mae if final_logged_validation_mae is not None and not np.isnan(final_logged_validation_mae) else None,
        'final_training_metrics_from_history_last_epoch': _filter_metrics_dict_for_log(logged_train_metrics_from_history),
        'final_validation_metrics_from_history_last_epoch': _filter_metrics_dict_for_log(logged_val_metrics_from_history),
        'encoder_plugin_params_logged': _filter_plugin_params_for_log(
            best_autoencoder_manager.encoder_plugin.get_debug_info()
            if best_autoencoder_manager and best_autoencoder_manager.encoder_plugin and hasattr(best_autoencoder_manager.encoder_plugin, 'get_debug_info') else {}
        ),
        'decoder_plugin_params_logged': _filter_plugin_params_for_log(
            best_autoencoder_manager.decoder_plugin.get_debug_info()
            if best_autoencoder_manager and best_autoencoder_manager.decoder_plugin and hasattr(best_autoencoder_manager.decoder_plugin, 'get_debug_info') else {}
        ),
        'key_config_parameters_logged': config_subset_for_debug # This subset will be sanitized by save_debug_info
    }

    if 'save_log' in config and config['save_log']:
        # save_debug_info now applies sanitize_dict_for_json internally
        save_debug_info(debug_info, config['save_log'])
    
    if 'remote_log' in config and config['remote_log']:
        # remote_log also applies sanitize_dict_for_json to its config_arg
        remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
    
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
    
    encoder_model_path = config['load_encoder']
    if not encoder_model_path or not os.path.exists(encoder_model_path):
        print(f"Encoder model path not found or not specified: {encoder_model_path}")
        return

    encoder_component_model = tf.keras.models.load_model(encoder_model_path, custom_objects=custom_objects, compile=False) # Changed
    print(f"CVAE Encoder component model loaded from {encoder_model_path}")

    x_input_file = config.get('x_test_file', config.get('x_validation_file')) 
    if not x_input_file: raise ValueError("No input file (x_test_file or x_validation_file) specified for encoder evaluation.")
    
    temp_eval_config = config.copy()
    temp_eval_config['x_train_file'] = x_input_file 
    temp_eval_config['y_train_file'] = x_input_file 
    temp_eval_config['x_validation_file'] = None 
    temp_eval_config['y_validation_file'] = None
    
    from app.plugin_loader import load_plugin 
    preprocessor_class = load_plugin(config['preprocessor_plugin'], 'preprocessor_plugins')
    eval_preprocessor = preprocessor_class()

    eval_datasets = eval_preprocessor.run_preprocessing(temp_eval_config)
    x_window_eval_data = eval_datasets.get("x_train") 

    if x_window_eval_data is None or x_window_eval_data.ndim != 3:
        raise ValueError(f"Evaluation data 'x_window_eval_data' from preprocessor is not correctly shaped or is None. Shape: {getattr(x_window_eval_data, 'shape', 'N/A')}")

    num_samples_eval = x_window_eval_data.shape[0]

    rnn_hidden_dim_eval = config.get('rnn_hidden_dim') 
    conditioning_dim_eval = config.get('conditioning_dim')
    if rnn_hidden_dim_eval is None or conditioning_dim_eval is None:
        raise ValueError("rnn_hidden_dim and conditioning_dim must be in config for encoder evaluation.")

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
    
    # Assuming encoder_component_model outputs [z_mean, z_log_var]
    if isinstance(encoded_outputs, list) and len(encoded_outputs) > 0:
        z_mean_data = encoded_outputs[0]
    elif isinstance(encoded_outputs, np.ndarray): # If only one output (e.g. just z_mean)
        z_mean_data = encoded_outputs
    else:
        raise ValueError(f"Unexpected output format from encoder component model: {type(encoded_outputs)}")

    print(f"Encoded z_mean shape: {z_mean_data.shape}")

    if config.get('evaluate_encoder'):
        output_filename = config['evaluate_encoder']
        encoded_df = pd.DataFrame(z_mean_data)
        encoded_df.columns = [f'latent_feature_{i}' for i in range(z_mean_data.shape[1])]
        
        eval_dates = eval_datasets.get("x_train_dates") 
        if eval_dates is not None and len(eval_dates) == len(encoded_df):
            encoded_df.index = pd.to_datetime(eval_dates)
            print(f"Added dates to encoded output from preprocessor.")
        
        write_csv(encoded_df, output_filename, index=eval_dates is not None) 
        print(f"Encoded z_mean data saved to {output_filename}")


def load_and_evaluate_decoder(config):
    decoder_model_path = config['load_decoder']
    if not decoder_model_path or not os.path.exists(decoder_model_path):
        print(f"Decoder model path not found or not specified: {decoder_model_path}")
        return
        
    decoder_component_model = tf.keras.models.load_model(decoder_model_path, compile=False) # Changed
    print(f"CVAE Decoder component model loaded from {decoder_model_path}")

    z_t_input_file = config.get('evaluate_encoder_output_for_decoder', config.get('evaluate_encoder')) 
    if not z_t_input_file:
        z_t_input_file = config.get('x_test_file_for_z_samples') 
        if not z_t_input_file:
            raise ValueError("No input file specified for z_t samples for decoder evaluation.")
    
    if not os.path.exists(z_t_input_file):
        raise FileNotFoundError(f"Input file for z_t samples not found: {z_t_input_file}")

    z_t_df = load_csv(file_path=z_t_input_file, headers=True, force_date=False) 
    z_t_data = z_t_df.to_numpy()

    num_samples_eval = z_t_data.shape[0]
    if z_t_data.shape[1] != config.get('latent_dim'):
         print(f"Warning: Loaded z_t data feature count ({z_t_data.shape[1]}) does not match config's latent_dim ({config.get('latent_dim')}). Ensure correct z_t input file.")
    
    rnn_hidden_dim_eval = config.get('rnn_hidden_dim')
    conditioning_dim_eval = config.get('conditioning_dim')
    if rnn_hidden_dim_eval is None or conditioning_dim_eval is None:
        raise ValueError("rnn_hidden_dim and conditioning_dim must be in config for decoder evaluation.")

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
        
        target_feature_names_eval = config.get('cvae_target_feature_names', ['OPEN', 'LOW', 'HIGH', 'vix_close', 'BC-BO', 'BH-BL']) # Default might need update if used often
        if len(target_feature_names_eval) == decoded_data.shape[1]:
            decoded_df.columns = target_feature_names_eval
        else:
            decoded_df.columns = [f'reconstructed_feature_{i}' for i in range(decoded_data.shape[1])]
            print(f"Warning: Number of target feature names ({len(target_feature_names_eval)}) "
                  f"does not match decoded data features ({decoded_data.shape[1]}). Using generic column names.")

        if isinstance(z_t_df.index, pd.DatetimeIndex):
            decoded_df.index = z_t_df.index
            print(f"Added dates to decoded output from z_t input file.")
        
        write_csv(decoded_df, output_filename, index=isinstance(z_t_df.index, pd.DatetimeIndex))
        print(f"Decoded data saved to {output_filename}")



