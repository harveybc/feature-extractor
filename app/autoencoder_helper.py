import numpy as npas tf
import pandas as pd # Added for DataFrame creation in evaluate functionsomething tf.keras doesn't cover
import time # Added for execution_timeber
import tensorflow as tf # Added for tf.print consistencyduceLROnPlateau, Callback # Changed to tensorflow.keras
from app.autoencoder_manager import AutoencoderManager
from app.data_handler import load_csv, write_csv
# from app.reconstruction import unwindow_data es
from app.config_handler import save_debug_info, remote_logprinted by EpochEndLogger
import os # Add os import for load_and_evaluate_encoder/decoder path checksacker") # Will be updated from logs['lr']
from tensorflow.keras.models import load_model # Changedat32, name="kl_beta_callback_tracker")
from tensorflow.keras.utils import plot_model # Changedes_wait_tracker")
es_patience_config_tracker = tf.Variable(0, dtype=tf.int32, name="es_patience_config_tracker")
es_best_value_tracker = tf.Variable(np.inf, dtype=tf.float32, name="es_best_value_tracker")
# This utility function might still be useful for a preprocessor plugin,cker")
# but not directly for creating the CVAE's per-step inputs in this file.op_patience_config_tracker")
def create_sliding_windows(data, window_size):
    """ers for loss components - RENAMED for clarity
    Create sliding windows for the entire dataset, returning a 3D array.name="mmd_total_component_tracker") # RENAMED from mmd_total
    r_loss_component_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="huber_loss_component_tracker") # RENAMED from huber_loss_tracker
    Args:_component_tracker = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="skew_loss_component_tracker") # RENAMED from skew_loss_tracker
        data (pd.DataFrame): Dataframe containing the time series features.ble=False, name="kurtosis_loss_component_tracker") # RENAMED from kurtosis_loss_tracker
        window_size (int): The length of the sliding window.f.float32, trainable=False, name="covariance_loss_component_tracker") # RENAMED from covariance_loss_tracker
    
    Returns:
        np.ndarray: Array with shape (num_samples, window_size, num_features).
    """
    data_array = data.to_numpy()sian kernel (biased estimator for MMD^2).
    num_features = data_array.shape[1], feature_dim)
    num_samples = data_array.shape[0] - window_size + 1
    """
    windows = np.zeros((num_samples, window_size, num_features))g graph construction or eagerly if possible
    if isinstance(sigma, (float, int)) and sigma <= 1e-6: 
    for i in range(num_samples): Warning: sigma is very small or zero. Setting to 1.0.")
        windows[i] = data_array[i:i+window_size]
    elif tf.is_tensor(sigma): # If sigma is a tensor, use tf.cond for the check
    return windows.cond(
            tf.less_equal(sigma, 1e-6),
            lambda: tf.constant(1.0, dtype=sigma.dtype),
def run_autoencoder_pipeline(config, encoder_plugin, decoder_plugin, preprocessor_plugin):
    start_time = time.time()
    
    tf.print("Loading/processing datasets via PreprocessorPlugin...")
    datasets = preprocessor_plugin.run_preprocessing(config)
    tf.print("PreprocessorPlugin finished.")
    if sample_size is not None:
    x_train_data = datasets.get("x_train") 
    x_val_data = datasets.get("x_val")
    feature_names_all = datasets.get("feature_names") 
        current_sample_size_x = tf.minimum(sample_size, batch_size_x)
    if x_train_data is None: # x_val_data can be None if no validation is performed
        raise ValueError("PreprocessorPlugin did not return 'x_train' data.")
    if feature_names_all is None:onal sampling
        raise ValueError("PreprocessorPlugin did not return 'feature_names'.")
    if not isinstance(feature_names_all, list):),
        raise TypeError(f"'feature_names' must be a list, got {type(feature_names_all)}")e_size_x]),
            lambda: tf.zeros([0, tf.shape(x)[-1]], dtype=x.dtype)
    target_feature_names = config.get('cvae_target_feature_names', ['OPEN', 'LOW', 'HIGH', 'vix_close', 'BC-BO', 'BH-BL'])
    tf.print(f"Using CVAE target features: {target_feature_names}")
    tf.print(f"Number of CVAE target features: {len(target_feature_names)}") # ADDED: Print count
            tf.greater(current_sample_size_y, 0),
    try:    lambda: tf.gather(y, tf.random.shuffle(tf.range(batch_size_y))[:current_sample_size_y]),
        target_indices = [feature_names_all.index(name) for name in target_feature_names]
    except ValueError as e:
        missing_feature = str(e).split("'")[1] 
        raise ValueError(CHECK ---
            f"One of the CVAE target features ('{missing_feature}') not found in 'feature_names' "
            f"provided by PreprocessorPlugin. Available features: {feature_names_all}"
        ) from ewise_sq_distances(a, b):
            a_sum_sq = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
    if x_train_data.ndim != 3: # Expect 3D (samples, window_size, features)
        raise ValueError(f"x_train_data is expected to be 3D (samples, window, features), but got shape {x_train_data.shape}")
            dist_sq = a_sum_sq + tf.transpose(b_sum_sq) - 2 * ab_dot
    y_train_targets = x_train_data[:, -1, target_indices] # RENAMED: from y_train_targets_6_features
    tf.print(f"Constructed y_train_targets with shape: {y_train_targets.shape}") # MODIFIED: Print statement
        k_xx_dist = pairwise_sq_distances(x_sample, x_sample)
    y_val_targets = None # RENAMED: from y_val_targets_6_features
    if x_val_data is not None:q_distances(x_sample, y_sample)
        if x_val_data.ndim != 3: # Expect 3D
            raise ValueError(f"x_val_data is expected to be 3D (samples, window, features), but got shape {x_val_data.shape}")
        y_val_targets = x_val_data[:, -1, target_indices] # RENAMED & Extract last stepis extremely small
        tf.print(f"Constructed y_val_targets with shape: {y_val_targets.shape}") # MODIFIED: Print statement
        if x_val_data.shape[0] != y_val_targets.shape[0]: # MODIFIED: Variable name
            raise ValueError(f"Sample count mismatch between x_val_data ({x_val_data.shape[0]}) and y_val_targets ({y_val_targets.shape[0]})") # MODIFIED
    else:_xy = tf.exp(-k_xy_dist / (2.0 * safe_sigma_sq))
        tf.print("No x_val_data provided by preprocessor. Validation will be skipped if not set in config later.")
        mean_k_xx = tf.reduce_mean(k_xx)
        mean_k_yy = tf.reduce_mean(k_yy)
    if not isinstance(y_train_targets, np.ndarray): # MODIFIED: Variable name
        raise TypeError(f"y_train_targets is type {type(y_train_targets)}, expected np.ndarray.") # MODIFIED
    if y_val_targets is not None and not isinstance(y_val_targets, np.ndarray): # MODIFIED: Variable name
        raise TypeError(f"y_val_targets is type {type(y_val_targets)}, expected np.ndarray.") # MODIFIED

    # y_targets should be 2D (samples, num_target_features)ma, "Shapes X_s, Y_s:", tf.shape(x_sample), tf.shape(y_sample),
    if y_train_targets.ndim != 2 or y_train_targets.shape[-1] != len(target_feature_names): # MODIFIED: Variable name
        raise ValueError(f"y_train_targets should be 2D with {len(target_feature_names)} features, but got shape {y_train_targets.shape}") # MODIFIED
    if y_val_targets is not None and (y_val_targets.ndim != 2 or y_val_targets.shape[-1] != len(target_feature_names)): # MODIFIED: Variable name
        raise ValueError(f"y_val_targets should be 2D with {len(target_feature_names)} features, but got shape {y_val_targets.shape}") # MODIFIED
        return tf.cond(
    if x_train_data.shape[0] != y_train_targets.shape[0]: # MODIFIED: Variable name
        raise ValueError(f"Sample count mismatch between x_train_data ({x_train_data.shape[0]}) and y_train_targets ({y_train_targets.shape[0]})") # MODIFIED
            lambda: mmd_val_calc
    # --- Populate necessary dimensions in config ---
    if 'window_size' not in config or not isinstance(config.get('window_size'), int) or config.get('window_size') <= 0:
        if x_train_data.ndim == 3: # Check if x_train_data is 3D
            config['window_size'] = x_train_data.shape[1] are empty. Returning 0.")
            tf.print(f"Derived 'window_size' from 3D preprocessed training data: {config['window_size']}")
        else: # Handle case where x_train_data is not 3D and window_size is not in config
            raise ValueError("x_train_data is not 3D and 'window_size' is not specified in config. Cannot derive window_size.")
    elif x_train_data.ndim == 3 and x_train_data.shape[1] != config['window_size']: # If 3D and mismatch
         tf.print(f"Warning: Mismatch between explicitly set config 'window_size' ({config['window_size']}) and "
               f"3D x_train_data.shape[1] ({x_train_data.shape[1]}). Using config 'window_size'.")
    elif x_train_data.ndim != 3: # If not 3D but window_size is in config
        tf.print(f"Warning: 'window_size' is set in config to {config['window_size']}, but x_train_data is not 3D (shape: {x_train_data.shape}). Proceeding with config 'window_size'. Ensure model handles this.")
    # --- END MODIFIED EMPTY CHECK ---
    
    if x_train_data.ndim == 3:
        config['num_features_input'] = x_train_data.shape[2]
    elif x_train_data.ndim == 2: # If preprocessor returns 2D (samples, features) and windowing is internal to model
        config['num_features_input'] = x_train_data.shape[1]
        tf.print(f"Warning: x_train_data is 2D. Assuming 'num_features_input' is x_train_data.shape[1]. Ensure model handles windowing if needed.")
    else: # Should have been caught by earlier x_train_data.ndim != 3 check
        raise ValueError(f"Cannot determine 'num_features_input'. x_train_data has unexpected shape: {x_train_data.shape}")
    tf.print(f"[DataPrep] Automatically set 'num_features_input': {config['num_features_input']}")
    def calculate_moment():
    config['num_features_output'] = y_train_targets.shape[1] # MODIFIED: num_features_output is from the target
    tf.print(f"[DataPrep] Automatically set 'num_features_output': {config['num_features_output']} from y_train_targets.shape[1]") # MODIFIED
    def return_zero_moment():
    required_dims = {stant(0.0, dtype=tf.float32)
        'rnn_hidden_dim': "RNN hidden state dimension for CVAE components.",
        'conditioning_dim': "Conditioning vector dimension for CVAE components.",
        'latent_dim': "Latent space dimension for CVAE."
    }   tf.math.less_equal(std_dev, 1e-9), # Check if std_dev is very close to zero
    for dim_key, description in required_dims.items():
        if dim_key not in config or not isinstance(config.get(dim_key), int) or config.get(dim_key) <= 0:
            raise ValueError(f"'{dim_key}' not found in config or is not a positive integer. {description} It's required.")
    # --- END MODIFIED CHECK ---
    num_train_samples = x_train_data.shape[0]
    h_context_train = datasets.get("h_train")
    if h_context_train is None:, y_pred, cfg):
        tf.print(f"Warning: 'h_train' not found in datasets. Generating zeros for h_context_train (shape: ({num_train_samples}, {config['rnn_hidden_dim']})).")
        h_context_train = np.zeros((num_train_samples, config['rnn_hidden_dim']), dtype=np.float32)
    elif h_context_train.shape != (num_train_samples, config['rnn_hidden_dim']):
        raise ValueError(f"Shape mismatch for h_context_train. Expected ({num_train_samples}, {config['rnn_hidden_dim']}), got {h_context_train.shape}")
# NEW WRAPPER FUNCTION - THIS IS THE ONE TO IMPORT
    conditions_t_train = datasets.get("cond_train")ig):
    if conditions_t_train is None:
        tf.print(f"Warning: 'cond_train' not found in datasets. Generating zeros for conditions_t_train (shape: ({num_train_samples}, {config['conditioning_dim']})).")
        conditions_t_train = np.zeros((num_train_samples, config['conditioning_dim']), dtype=np.float32)
    elif conditions_t_train.shape != (num_train_samples, config['conditioning_dim']):
        raise ValueError(f"Shape mismatch for conditions_t_train. Expected ({num_train_samples}, {config['conditioning_dim']}), got {conditions_t_train.shape}")
        config_to_use = outer_config 
    cvae_train_inputs = [x_train_data, h_context_train, conditions_t_train]
    cvae_train_targets = y_train_targets # MODIFIED: Variable name, tf.float32)
        recon_pred = tf.cast(y_pred_recon_tensor, tf.float32)
    # Validation data setup
    if x_val_data is not None and y_val_targets is not None: # MODIFIED: Variable name
        num_val_samples = x_val_data.shape[0]gma', 1.0)
        h_context_val = datasets.get("h_val")mmd_sample_size', None)
        if h_context_val is None:
            tf.print(f"Warning: 'h_val' not found in datasets. Generating zeros for h_context_val (shape: ({num_val_samples}, {config['rnn_hidden_dim']})).")
            h_context_val = np.zeros((num_val_samples, config['rnn_hidden_dim']), dtype=np.float32)
        elif h_context_val.shape != (num_val_samples, config['rnn_hidden_dim']):
            raise ValueError(f"Shape mismatch for h_context_val. Expected ({num_val_samples}, {config['rnn_hidden_dim']}), got {h_context_val.shape}")

        conditions_t_val = datasets.get("cond_val")
        if conditions_t_val is None:Huber(delta=huber_delta)
            tf.print(f"Warning: 'cond_val' not found in datasets. Generating zeros for conditions_t_val (shape: ({num_val_samples}, {config['conditioning_dim']})).")
            conditions_t_val = np.zeros((num_val_samples, config['conditioning_dim']), dtype=np.float32)
        elif conditions_t_val.shape != (num_val_samples, config['conditioning_dim']):
            raise ValueError(f"Shape mismatch for conditions_t_val. Expected ({num_val_samples}, {config['conditioning_dim']}), got {conditions_t_val.shape}")
        if mmd_weight > 0:
        config['cvae_val_inputs'] = {ual_reconstruction_target, recon_pred, sigma=mmd_sigma, sample_size=mmd_sample_size)
            'x_window': x_val_data,n(mmd_val) # Use RENAMED tracker
            'h_context': h_context_val,mmd_val
            'conditions_t': conditions_t_val
        }   mmd_total_tracker.assign(0.0) # Use RENAMED tracker
        config['cvae_val_targets'] = y_val_targets # MODIFIED: Variable name
        tf.print(f"[data_processor] Added cvae_val_inputs and cvae_val_targets to config from preprocessor output.")
    else:   skew_true = calculate_standardized_moment(tf.reshape(actual_reconstruction_target, [-1]), 3)
        tf.print("[data_processor] Validation data (x_val_data or y_val_targets from preprocessor) is None. " # MODIFIED
                 "Training will proceed without validation unless 'cvae_val_inputs' and 'cvae_val_targets' are already in config.")
        if 'cvae_val_inputs' not in config or 'cvae_val_targets' not in config:cker
            config.pop('cvae_val_inputs', None) s_val
            config.pop('cvae_val_targets', None)
            tf.print("[data_processor] No validation data found in preprocessor output or existing config.")

        if kurtosis_weight > 0:
    initial_latent_dim = config.get('initial_latent_dim', config['latent_dim']) uction_target, [-1]), 4)
    step_size_latent = config.get('step_size_latent', 8) reshape(recon_pred, [-1]), 4)
    threshold_error = config.get('threshold_error', 0.1) # MAE threshold
    epochs = config.get('epochs', 50)racker.assign(kurt_loss_val) # Use RENAMED tracker
    batch_size = config.get('batch_size', 128)rt_loss_val
    incremental_search = config.get('incremental_search', False) 
            kurtosis_loss_component_tracker.assign(0.0) # Use RENAMED tracker
    current_latent_dim = initial_latent_dim
    best_val_mae = float('inf')
    best_latent_dim = current_latent_dimss_calc(actual_reconstruction_target, recon_pred, config_to_use) 
    best_autoencoder_manager = Nonent_tracker.assign(cov_loss_val) # Use RENAMED tracker
            total_loss += cov_weight * cov_loss_val
    # This key should now be 'reconstruction_out_custom_recon_mae' 
    # because autoencoder_manager.py is using MeanAbsoluteError(name='custom_recon_mae').
    # Keras prepends the output name 'reconstruction_out'.
    # If 'mae' (string) is used in compile, key is 'reconstruction_out_mae'.
    # If tf.keras.metrics.MeanAbsoluteError() instance is used (without name), key is 'reconstruction_out_mean_absolute_error'.
    # NEW: With dedicated output 'reconstruction_out_for_mae_calc' and MeanAbsoluteError metric:
    expected_mae_key = 'reconstruction_out_for_mae_calc_mean_absolute_error' # UPDATED EXPECTED KEY
    expected_val_mae_key = f'val_{expected_mae_key}' # Keras prepends 'val_''
# if 'get_reconstruction_and_stats_loss_fn' is now the primary way to get the loss function.
    while True:
        config['latent_dim'] = current_latent_dim n_tensor, y_pred_recon_tensor, config=None):
        tf.print(f"Training CVAE with latent_dim: {current_latent_dim}")ENTED IF USING THE WRAPPER ABOVE)
         config is None: config = {} 
        autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
        autoencoder_manager.build_autoencoder(config) 
        
        autoencoder_manager.train_autoencoder(
            data_inputs=cvae_train_inputs,tly used by AutoencoderManager._compile_model
            data_targets=cvae_train_targets,
            epochs=epochs,re was 'calculate_mae_for_reconstruction'.
            batch_size=batch_size,uction(y_true_tensor, y_pred_tensor): 
            config=config te_mae_for_reconstruction] Inside MAE metric function.")
        )f.print("y_true_tensor shape:", tf.shape(y_true_tensor), "y_true_tensor dtype:", y_true_tensor.dtype)
        tf.print("y_pred_tensor shape:", tf.shape(y_pred_tensor), "y_pred_tensor dtype:", y_pred_tensor.dtype)
        train_eval_results = autoencoder_manager.evaluate(cvae_train_inputs, cvae_train_targets, "Training", config)
        # Print a few sample values
        training_mae = float('nan')ple (first 5):", y_true_tensor[0,:5] if tf.shape(y_true_tensor)[0] > 0 else "N/A")
        if train_eval_results and expected_mae_key in train_eval_results:f tf.shape(y_pred_tensor)[0] > 0 else "N/A")
            training_mae = train_eval_results[expected_mae_key]
        yt_recon = tf.cast(y_true_tensor, tf.float32)
        if np.isnan(training_mae):tensor, tf.float32)
            tf.print(f"CRITICAL ERROR: Training MAE ('{expected_mae_key}') is NaN or key not found in evaluation results!")
            tf.print(f"Evaluation results for Training (train_eval_results):")
            if isinstance(train_eval_results, dict):
                for k_iter, v_iter in train_eval_results.items():s_nan(yt_recon)))
                    tf.print(f"  Key: '{k_iter}', Value: {v_iter}, Type: {type(v_iter)}")
            else:"Any NaNs in yp_recon:", tf.reduce_any(tf.math.is_nan(yp_recon)))
                tf.print(f"  train_eval_results is not a dict: {train_eval_results}")
            
            compiled_metrics_names_str = "N/A"
            if hasattr(autoencoder_manager, 'autoencoder_model') and autoencoder_manager.autoencoder_model and hasattr(autoencoder_manager.autoencoder_model, 'metrics_names'):
                 compiled_metrics_names_str = str(autoencoder_manager.autoencoder_model.metrics_names)
            tf.print(f"Model's compiled metrics_names attribute: {compiled_metrics_names_str}")
            tf.print(f"Model's actual metric objects (name and class):")
            if hasattr(autoencoder_manager, 'autoencoder_model') and autoencoder_manager.autoencoder_model and hasattr(autoencoder_manager.autoencoder_model, 'metrics'):
                for m_obj in autoencoder_manager.autoencoder_model.metrics:diff)))
                    tf.print(f"  - {m_obj.name} ({m_obj.__class__.__name__})")
            else: = tf.reduce_mean(abs_diff)
                tf.print("  Could not retrieve model.metrics objects.")
            raise ValueError(f"Training MAE ('{expected_mae_key}') is missing or NaN. Stopping execution. Results: {train_eval_results}. Compiled metrics: {compiled_metrics_names_str}")
        # Check for NaNs/Infs in final MAE
        tf.print(f"Training MAE with latent_dim {current_latent_dim}: {training_mae:.4f} (extracted with key: {expected_mae_key})")
        tf.print(f"Full Training Evaluation Results: {train_eval_results}")
        
        validation_mae = float('nan') aN to avoid issues down the line, though Keras might handle it.
        if config.get('cvae_val_inputs') and config.get('cvae_val_targets') is not None: mae_value)
            eval_val_inputs_list = [
                config['cvae_val_inputs']['x_window'],
                config['cvae_val_inputs']['h_context'],on] 
                config['cvae_val_inputs']['conditions_t']rics_to_return[0].__name__)
            ]trics_to_return
            eval_val_targets_array = config['cvae_val_targets']
            toppingWithPatienceCounter(tf.keras.callbacks.EarlyStopping): # Changed to tf.keras
            val_eval_results = autoencoder_manager.evaluate(eval_val_inputs_list, eval_val_targets_array, "Validation", config)
            r().__init__(**kwargs)
            # Use expected_mae_key directly, as model.evaluate() does not add 'val_' prefix to keys in its returned dict
            if val_eval_results and expected_mae_key in val_eval_results: # CHANGED: Use expected_mae_key
                validation_mae = val_eval_results[expected_mae_key]       # CHANGED: Use expected_mae_key

            if np.isnan(validation_mae):one):
                # The expected_val_mae_key is what we *would* look for if Keras prefixed it, 
                # but for the error message, it's more informative to state what was attempted.
                # However, the actual lookup should use expected_mae_key.
                # For clarity in the error message if it still fails for other reasons, we can keep expected_val_mae_key here.
                tf.print(f"CRITICAL ERROR: Validation MAE (expected key pattern: '{expected_val_mae_key}', actual lookup key: '{expected_mae_key}') is NaN or key not found in evaluation results!") # MODIFIED ERROR MSG
                tf.print(f"Evaluation results for Validation (val_eval_results):")
                if isinstance(val_eval_results, dict):dLogger will handle display
                    for k_iter, v_iter in val_eval_results.items():
                        tf.print(f"  Key: '{k_iter}', Value: {v_iter}, Type: {type(v_iter)}") # Added print for value and type
                else:, **kwargs):
                    tf.print(f"  val_eval_results is not a dict: {val_eval_results}")
        # Store configured patience in the global tracker
                compiled_metrics_names_str = "N/A"
                if hasattr(autoencoder_manager, 'autoencoder_model') and autoencoder_manager.autoencoder_model and hasattr(autoencoder_manager.autoencoder_model, 'metrics_names'):
                    compiled_metrics_names_str = str(autoencoder_manager.autoencoder_model.metrics_names)
                tf.print(f"Model's compiled metrics_names attribute: {compiled_metrics_names_str}")
                tf.print(f"Model's actual metric objects (name and class):")e it
                if hasattr(autoencoder_manager, 'autoencoder_model') and autoencoder_manager.autoencoder_model and hasattr(autoencoder_manager.autoencoder_model, 'metrics'):
                    for m_obj in autoencoder_manager.autoencoder_model.metrics:gs['lr']
                        tf.print(f"  - {m_obj.name} ({m_obj.__class__.__name__})") # Added print for metric object
                else:e logs['lr'] (added by ReduceLROnPlateau base class) is sufficient.
                    tf.print("  Could not retrieve model.metrics objects.")c
                # The error message should reflect the key that was *intended* to be found for validation.
                raise ValueError(f"Validation MAE ('{expected_val_mae_key}' pattern, actual lookup with '{expected_mae_key}') is missing or NaN. Stopping execution. Results: {val_eval_results}. Compiled metrics: {compiled_metrics_names_str}") # MODIFIED ERROR MSG
            rlrop_wait_tracker.assign(self.wait)
            tf.print(f"Validation MAE with latent_dim {current_latent_dim}: {validation_mae:.4f} (extracted with key: {expected_mae_key})") # CHANGED: Use expected_mae_key
            tf.print(f"Full Validation Evaluation Results: {val_eval_results}")
        else:
            tf.print(f"Validation data not configured for latent_dim {current_latent_dim}. Skipping validation MAE comparison for search.")
        __init__(self, kl_beta_start, kl_beta_end, anneal_epochs, 
        if not np.isnan(validation_mae) and validation_mae < best_val_mae:verbose=0): 
            best_val_mae = validation_mae__init__()
            best_latent_dim = current_latent_dim
            best_autoencoder_manager = autoencoder_manager 
            tf.print(f"New best Validation MAE: {best_val_mae:.4f} with latent_dim: {best_latent_dim}")
        elif np.isnan(validation_mae) and best_autoencoder_manager is None: 
            best_latent_dim = current_latent_dim
            best_autoencoder_manager = autoencoder_managerific prints from this cb are needed
            tf.print(f"No validation MAE. Storing model for latent_dim: {current_latent_dim} as current best.")

    def on_epoch_begin(self, epoch, logs=None):
        if not incremental_search:chs:
            tf.print("Incremental search disabled. Completing after one iteration.")* (epoch / float(self.anneal_epochs))
            if best_autoencoder_manager is None: best_autoencoder_manager = autoencoder_manager 
            break 
            self.current_kl_beta.assign(self.kl_beta_end)
        if not np.isnan(validation_mae) and validation_mae <= threshold_error:
            tf.print(f"Threshold MAE ({threshold_error}) met or improved. Optimal latent_dim likely found: {current_latent_dim}")
            break
        else: # Check if validation_mae is NaN before deciding to increment latent_dim
            if np.isnan(validation_mae) and not config.get('allow_search_without_val_mae', False): # Add a config flag if you want to allow search without val_mae
                tf.print(f"Validation MAE is NaN and search without val_mae is not allowed. Stopping search at latent_dim {current_latent_dim}.")
                if best_autoencoder_manager is None: best_autoencoder_manager = autoencoder_manager # Ensure a model is selected
                break
                target_kl_layer = self.model.get_layer(self.layer_name)
            current_latent_dim += step_size_latent
            if current_latent_dim > config.get('max_latent_dim', 256):int for setup is okay
                tf.print(f"Reached max_latent_dim ({config.get('max_latent_dim', 256)}). Stopping search.").")
                break
            if current_latent_dim <= 0: 
                tf.print(f"Latent dimension became non-positive ({current_latent_dim}). Stopping search.")
                breakt_kl_layer.kl_beta.assign(self.current_kl_beta)
            elif hasattr(target_kl_layer, 'kl_beta'): 
    if best_autoencoder_manager is None: lematic if kl_beta is not a tf.Variable.
        if autoencoder_manager: sure KLDivergenceLayer.kl_beta is always a tf.Variable.
            tf.print("Warning: best_autoencoder_manager was not set. Using the last trained model.")
            best_autoencoder_manager = autoencoder_manager
            best_latent_dim = config.get('latent_dim') 
        else:      tf.print(f"\nKLAnnealingCallback: Layer '{target_kl_layer.name}' does not have 'kl_beta' attribute or it's not assignable.")
            raise RuntimeError("Autoencoder training loop did not run or failed to select a model.")
            if self.verbose > 0 and epoch == 0: 
    best_val_mae_str = f"{best_val_mae:.4f}" if not np.isnan(best_val_mae) else "N/A"ta will not be annealed by this callback.")
    tf.print(f"Final selected latent_dim: {best_latent_dim} with Best Validation MAE: {best_val_mae_str}")
    def on_epoch_end(self, epoch, logs=None):
    config['latent_dim'] = best_latent_dim 
            logs['kl_beta_val'] = self.current_kl_beta.numpy() # Add to logs, use a distinct key
    encoder_model_filename = config.get('save_encoder', 'encoder_model.keras').replace(".keras", f"_ld{best_latent_dim}.keras").replace(".h5", f"_ld{best_latent_dim}.keras")
    decoder_model_filename = config.get('save_decoder', 'decoder_model.keras').replace(".keras", f"_ld{best_latent_dim}.keras").replace(".h5", f"_ld{best_latent_dim}.keras")
    w callback for consolidated epoch-end logging
    if best_autoencoder_manager.encoder_plugin:k): # Changed to tf.keras
        best_autoencoder_manager.save_encoder(encoder_model_filename) 
        tf.print(f"Saved best encoder component model to {encoder_model_filename}")
    if best_autoencoder_manager.decoder_plugin:
        best_autoencoder_manager.save_decoder(decoder_model_filename) 
        tf.print(f"Saved best decoder component model to {decoder_model_filename}")
        if 'loss' in logs: log_items.append(f"loss: {logs['loss']:.4f}")
    full_model_save_path_template = config.get('model_save_path', None)
    if full_model_save_path_template:put
        full_model_filename = full_model_save_path_template.replace(".keras", f"_ld{best_latent_dim}.keras").replace(".h5", f"_ld{best_latent_dim}.keras")
        best_autoencoder_manager.save_model(full_model_filename) 
        tf.print(f"Saved best full CVAE model to {full_model_filename}")
        # else: MAE not found in logs, will not be printed for this epoch

    end_time = time.time()ogs: log_items.append(f"val_loss: {logs['val_loss']:.4f}")
    execution_time = end_time - start_time
        val_mae_key = f"val_{mae_key}" 
    final_train_eval_results = best_autoencoder_manager.evaluate(cvae_train_inputs, cvae_train_targets, "Final Training", config)
    final_training_mae = float('nan')e: {logs[val_mae_key]:.4f}")
    if final_train_eval_results and expected_mae_key in final_train_eval_results:
        final_training_mae = final_train_eval_results[expected_mae_key]
        # Learning rate
    if np.isnan(final_training_mae):
        tf.print(f"CRITICAL ERROR: Final Training MAE ('{expected_mae_key}') is NaN or key not found!")
        tf.print(f"Final Training Evaluation Results: {final_train_eval_results}")
        compiled_metrics_names_str = "N/A"r') and hasattr(self.model.optimizer, 'learning_rate'):
        if hasattr(best_autoencoder_manager, 'autoencoder_model') and best_autoencoder_manager.autoencoder_model and hasattr(best_autoencoder_manager.autoencoder_model, 'metrics_names'):
            compiled_metrics_names_str = str(best_autoencoder_manager.autoencoder_model.metrics_names)
        tf.print(f"Model's compiled metrics names: {compiled_metrics_names_str}")
        raise ValueError(f"Final Training MAE ('{expected_mae_key}') is missing or NaN. Results: {final_train_eval_results}. Compiled metrics: {compiled_metrics_names_str}")
    tf.print(f"Final Training MAE (best model): {final_training_mae:.4f} (key: {expected_mae_key})")
                     # Get current step, might need a global step counter if not using iterations_per_epoch
                    current_step = self.model.optimizer.iterations.numpy() # Get current iterations
    final_val_eval_results = Nonel_str = f"{lr_val_obj(current_step).numpy():.7f}"
    final_validation_mae = float('nan') direct float value
    if config.get('cvae_val_inputs') and config.get('cvae_val_targets') is not None:7f}"
        final_eval_val_inputs_list = [
            config['cvae_val_inputs']['x_window'],e LR directly: {e}") # Optional debug
            config['cvae_val_inputs']['h_context'],
            config['cvae_val_inputs']['conditions_t']
        ]
        final_eval_val_targets_array = config['cvae_val_targets']lback)
        final_val_eval_results = best_autoencoder_manager.evaluate(final_eval_val_inputs_list, final_eval_val_targets_array, "Final Validation", config)
        # Use expected_mae_key for results from model.evaluate()
        if final_val_eval_results and expected_mae_key in final_val_eval_results: # CHANGED: Use expected_mae_key
            final_validation_mae = final_val_eval_results[expected_mae_key]       # CHANGED: Use expected_mae_key
        log_items.append(f"mmd_c: {mmd_total_tracker.numpy():.4f}") # RENAMED
        if np.isnan(final_validation_mae): # Check if NaN after attempting to extractMED
            # For the error message, expected_val_mae_key shows the pattern we'd expect from 'fit' logs
            tf.print(f"CRITICAL ERROR: Final Validation MAE (expected key pattern from fit logs: '{expected_val_mae_key}', actual lookup key for evaluate: '{expected_mae_key}') is NaN or key not found!")
            tf.print(f"Final Validation Evaluation Results: {final_val_eval_results}")
            compiled_metrics_names_str = "N/A"
            if hasattr(best_autoencoder_manager, 'autoencoder_model') and best_autoencoder_manager.autoencoder_model and hasattr(best_autoencoder_manager.autoencoder_model, 'metrics_names'):
                compiled_metrics_names_str = str(best_autoencoder_manager.autoencoder_model.metrics_names)
            tf.print(f"Model's compiled metrics names: {compiled_metrics_names_str}")
            raise ValueError(f"Final Validation MAE (expected key pattern from fit logs: '{expected_val_mae_key}', actual lookup key for evaluate: '{expected_mae_key}') is missing or NaN. Results: {final_val_eval_results}. Compiled metrics: {compiled_metrics_names_str}")
            log_items.append(f"ES_wait: {es_wait_tracker.numpy()}/{es_patience_val}")
    final_validation_mae_str = f"{final_validation_mae:.4f}" if not np.isnan(final_validation_mae) else "N/A"
    # When printing, use the non-prefixed key for clarity as it's from evaluate()
    tf.print(f"Final Validation MAE (best model): {final_validation_mae_str} (key from evaluate(): {expected_mae_key})")
        rlrop_patience_val = rlrop_patience_config_tracker.numpy()
        if rlrop_patience_val > 0: # Only print if RLROP is active (patience configured)
    debug_info = {ems.append(f"RLROP_wait: {rlrop_wait_tracker.numpy()}/{rlrop_patience_val}")
        'execution_time_seconds': execution_time,
        'best_latent_dim': best_latent_dim,dard Python print        'encoder_plugin_params': best_autoencoder_manager.encoder_plugin.get_debug_info() if best_autoencoder_manager.encoder_plugin and hasattr(best_autoencoder_manager.encoder_plugin, 'get_debug_info') else None,        'decoder_plugin_params': best_autoencoder_manager.decoder_plugin.get_debug_info() if best_autoencoder_manager.decoder_plugin and hasattr(best_autoencoder_manager.decoder_plugin, 'get_debug_info') else None,        'final_validation_mae': final_validation_mae if not np.isnan(final_validation_mae) else None, # Store the float or None        'final_training_mae': final_training_mae if not np.isnan(final_training_mae) else None, # Store the float or None        'final_validation_metrics': final_val_eval_results,        'final_training_metrics': final_train_eval_results,        'config_used': {k: v for k, v in config.items() if not isinstance(v, np.ndarray)}     }    if 'save_log' in config and config['save_log']:        save_debug_info(debug_info, config['save_log'])        tf.print(f"Debug info saved to {config['save_log']}.")        if 'remote_log' in config and config['remote_log']:        remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])        tf.print(f"Debug info saved to {config['remote_log']}.")        model_plot_file_template = config.get('model_plot_file', None)    if model_plot_file_template and best_autoencoder_manager.autoencoder_model:        model_plot_filename = model_plot_file_template.replace(".png", f"_ld{best_latent_dim}.png")        try:             plot_model(best_autoencoder_manager.autoencoder_model, to_file=model_plot_filename, show_shapes=True, show_layer_names=True, expand_nested=True)            tf.print(f"Model plot saved to {model_plot_filename}")        except Exception as e:             tf.print(f"Could not plot model: {e}")            tf.print(f"Pipeline execution time: {execution_time:.2f} seconds")def load_and_evaluate_encoder(config):    custom_objects = {}         encoder_model_path = config['load_encoder']    if not encoder_model_path or not os.path.exists(encoder_model_path):        print(f"Encoder model path not found or not specified: {encoder_model_path}")        return    encoder_component_model = tf.keras.models.load_model(encoder_model_path, custom_objects=custom_objects, compile=False) # Changed    print(f"CVAE Encoder component model loaded from {encoder_model_path}")    x_input_file = config.get('x_test_file', config.get('x_validation_file'))     if not x_input_file: raise ValueError("No input file (x_test_file or x_validation_file) specified for encoder evaluation.")        temp_eval_config = config.copy()    temp_eval_config['x_train_file'] = x_input_file     temp_eval_config['y_train_file'] = x_input_file     temp_eval_config['x_validation_file'] = None     temp_eval_config['y_validation_file'] = None        from app.plugin_loader import load_plugin     preprocessor_class = load_plugin(config['preprocessor_plugin'], 'preprocessor_plugins')    eval_preprocessor = preprocessor_class()    eval_datasets = eval_preprocessor.run_preprocessing(temp_eval_config)    x_window_eval_data = eval_datasets.get("x_train")     if x_window_eval_data is None or x_window_eval_data.ndim != 3:        raise ValueError(f"Evaluation data 'x_window_eval_data' from preprocessor is not correctly shaped or is None. Shape: {getattr(x_window_eval_data, 'shape', 'N/A')}")    num_samples_eval = x_window_eval_data.shape[0]    rnn_hidden_dim_eval = config.get('rnn_hidden_dim')     conditioning_dim_eval = config.get('conditioning_dim')    if rnn_hidden_dim_eval is None or conditioning_dim_eval is None:        raise ValueError("rnn_hidden_dim and conditioning_dim must be in config for encoder evaluation.")    h_context_eval = eval_datasets.get("h_train")    if h_context_eval is None:        print(f"Generating placeholder h_context for encoder evaluation (shape: ({num_samples_eval}, {rnn_hidden_dim_eval})).")        h_context_eval = np.zeros((num_samples_eval, rnn_hidden_dim_eval), dtype=np.float32)    elif h_context_eval.shape[0] != num_samples_eval or h_context_eval.shape[1] != rnn_hidden_dim_eval:        raise ValueError(f"Shape mismatch for eval h_context. Expected ({num_samples_eval}, {rnn_hidden_dim_eval}), got {h_context_eval.shape}")    conditions_t_eval = eval_datasets.get("cond_train")    if conditions_t_eval is None:        print(f"Generating placeholder conditions_t for encoder evaluation (shape: ({num_samples_eval}, {conditioning_dim_eval})).")        conditions_t_eval = np.zeros((num_samples_eval, conditioning_dim_eval), dtype=np.float32)    elif conditions_t_eval.shape[0] != num_samples_eval or conditions_t_eval.shape[1] != conditioning_dim_eval:        raise ValueError(f"Shape mismatch for eval conditions_t. Expected ({num_samples_eval}, {conditioning_dim_eval}), got {conditions_t_eval.shape}")        encoder_inputs_eval = [x_window_eval_data, h_context_eval, conditions_t_eval]    print(f"Encoding data. Input shapes: x_window: {encoder_inputs_eval[0].shape}, h_context: {encoder_inputs_eval[1].shape}, conditions_t: {encoder_inputs_eval[2].shape}")    encoded_outputs = encoder_component_model.predict(encoder_inputs_eval, verbose=1)        # Assuming encoder_component_model outputs [z_mean, z_log_var]    if isinstance(encoded_outputs, list) and len(encoded_outputs) > 0:        z_mean_data = encoded_outputs[0]    elif isinstance(encoded_outputs, np.ndarray): # If only one output (e.g. just z_mean)        z_mean_data = encoded_outputs    else:        raise ValueError(f"Unexpected output format from encoder component model: {type(encoded_outputs)}")    print(f"Encoded z_mean shape: {z_mean_data.shape}")    if config.get('evaluate_encoder'):        output_filename = config['evaluate_encoder']        encoded_df = pd.DataFrame(z_mean_data)        encoded_df.columns = [f'latent_feature_{i}' for i in range(z_mean_data.shape[1])]                eval_dates = eval_datasets.get("x_train_dates")         if eval_dates is not None and len(eval_dates) == len(encoded_df):            encoded_df.index = pd.to_datetime(eval_dates)            print(f"Added dates to encoded output from preprocessor.")                write_csv(encoded_df, output_filename, index=eval_dates is not None)         print(f"Encoded z_mean data saved to {output_filename}")def load_and_evaluate_decoder(config):    decoder_model_path = config['load_decoder']    if not decoder_model_path or not os.path.exists(decoder_model_path):        print(f"Decoder model path not found or not specified: {decoder_model_path}")        return            decoder_component_model = tf.keras.models.load_model(decoder_model_path, compile=False) # Changed    print(f"CVAE Decoder component model loaded from {decoder_model_path}")    z_t_input_file = config.get('evaluate_encoder_output_for_decoder', config.get('evaluate_encoder'))     if not z_t_input_file:        z_t_input_file = config.get('x_test_file_for_z_samples')         if not z_t_input_file:            raise ValueError("No input file specified for z_t samples for decoder evaluation.")        if not os.path.exists(z_t_input_file):        raise FileNotFoundError(f"Input file for z_t samples not found: {z_t_input_file}")    z_t_df = load_csv(file_path=z_t_input_file, headers=True, force_date=False)     z_t_data = z_t_df.to_numpy()    num_samples_eval = z_t_data.shape[0]    if z_t_data.shape[1] != config.get('latent_dim'):         print(f"Warning: Loaded z_t data feature count ({z_t_data.shape[1]}) does not match config's latent_dim ({config.get('latent_dim')}). Ensure correct z_t input file.")        rnn_hidden_dim_eval = config.get('rnn_hidden_dim')    conditioning_dim_eval = config.get('conditioning_dim')    if rnn_hidden_dim_eval is None or conditioning_dim_eval is None:        raise ValueError("rnn_hidden_dim and conditioning_dim must be in config for decoder evaluation.")    print(f"Generating placeholder h_context and conditions_t for decoder evaluation.")    h_context_eval = np.zeros((num_samples_eval, rnn_hidden_dim_eval), dtype=np.float32)    conditions_t_eval = np.zeros((num_samples_eval, conditioning_dim_eval), dtype=np.float32)    decoder_inputs_eval = [z_t_data, h_context_eval, conditions_t_eval]    print(f"Decoding data. Input shapes: z_t: {decoder_inputs_eval[0].shape}, h_context: {decoder_inputs_eval[1].shape}, conditions_t: {decoder_inputs_eval[2].shape}")    decoded_data = decoder_component_model.predict(decoder_inputs_eval, verbose=1)    print(f"Decoded data (reconstructed x_prime_t) shape: {decoded_data.shape}")    if config.get('evaluate_decoder'):        output_filename = config['evaluate_decoder']        decoded_df = pd.DataFrame(decoded_data)                target_feature_names_eval = config.get('cvae_target_feature_names', ['OPEN', 'LOW', 'HIGH', 'vix_close', 'BC-BO', 'BH-BL']) # Default might need update if used often        if len(target_feature_names_eval) == decoded_data.shape[1]:            decoded_df.columns = target_feature_names_eval        else:            decoded_df.columns = [f'reconstructed_feature_{i}' for i in range(decoded_data.shape[1])]            print(f"Warning: Number of target feature names ({len(target_feature_names_eval)}) "                  f"does not match decoded data features ({decoded_data.shape[1]}). Using generic column names.")        if isinstance(z_t_df.index, pd.DatetimeIndex):            decoded_df.index = z_t_df.index            print(f"Added dates to decoded output from z_t input file.")                write_csv(decoded_df, output_filename, index=isinstance(z_t_df.index, pd.DatetimeIndex))        print(f"Decoded data saved to {output_filename}")