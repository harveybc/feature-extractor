import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from app.autoencoder_manager import AutoencoderManager
from app.data_handler import load_csv, write_csv
# from app.reconstruction import unwindow_data # May not be relevant for per-step CVAE output
from app.config_handler import save_debug_info, remote_log
from keras.models import load_model # Removed Sequential, Model as they are not directly used here

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
    
    print("Loading/processing datasets via PreprocessorPlugin...")
    # IMPORTANT: Assume preprocessor_plugin.run_preprocessing(config) returns a dictionary
    # where 'x_train' and 'x_val' are the primary data sequences (x_t values).
    # Shape for x_train/x_val should be (num_samples, x_feature_dim)
    # It might also return 'h_train', 'h_val' (for h_context) and 'cond_train', 'cond_val' (for conditions_t)
    # or these need to be generated/derived.
    datasets = preprocessor_plugin.run_preprocessing(config)
    print("PreprocessorPlugin finished.")

    x_train_data = datasets.get("x_train") 
    x_val_data = datasets.get("x_val")
    feature_names_all = datasets.get("feature_names") # Get all feature names

    if x_train_data is None or x_val_data is None:
        raise ValueError("PreprocessorPlugin did not return 'x_train' or 'x_val' data.")
    if feature_names_all is None:
        raise ValueError("PreprocessorPlugin did not return 'feature_names'.")
    if not isinstance(feature_names_all, list):
        raise TypeError(f"'feature_names' must be a list, got {type(feature_names_all)}")

    # Define the 6 target feature names for the CVAE
    # IMPORTANT: Ensure these names EXACTLY match the names in 'feature_names_all'
    # Assuming 'BH-BL' from preprocessor output corresponds to your 'BHIGH-BLOW' target
    target_feature_names = ['OPEN', 'LOW', 'HIGH', 'CLOSE', 'BC-BO', 'BH-BL'] 
    
    # Find indices of these target features
    try:
        target_indices = [feature_names_all.index(name) for name in target_feature_names]
    except ValueError as e:
        missing_feature = str(e).split("'")[1] # Extract missing feature name
        raise ValueError(
            f"One of the CVAE target features ('{missing_feature}') not found in 'feature_names' "
            f"provided by PreprocessorPlugin. Available features: {feature_names_all}"
        ) from e

    # Extract the 6 target features from the LAST time step of each window in x_train_data and x_val_data
    # x_train_data shape: (num_samples, window_size, num_all_features)
    # We want y_train_targets_6_features shape: (num_samples, 6)
    if len(x_train_data.shape) != 3:
        raise ValueError(f"x_train_data is expected to be 3D (samples, window, features), but got shape {x_train_data.shape}")
    if len(x_val_data.shape) != 3:
        raise ValueError(f"x_val_data is expected to be 3D (samples, window, features), but got shape {x_val_data.shape}")

    y_train_targets_6_features = x_train_data[:, -1, target_indices]
    y_val_targets_6_features = x_val_data[:, -1, target_indices]
    
    print(f"Constructed y_train_targets_6_features with shape: {y_train_targets_6_features.shape}")
    print(f"Constructed y_val_targets_6_features with shape: {y_val_targets_6_features.shape}")

    # The following checks should now pass if the above logic is correct
    if y_train_targets_6_features is None or y_val_targets_6_features is None:
        # This should not be hit if extraction was successful, but keep as a safeguard
        raise ValueError("Failed to construct 'y_train_cvae_target' or 'y_val_cvae_target' data (6 features).")

    # Add a type check to ensure it's a NumPy array before accessing .shape
    if not isinstance(y_train_targets_6_features, np.ndarray):
        raise TypeError(f"y_train_cvae_target (expected 6-feature array) is type {type(y_train_targets_6_features)}, expected np.ndarray.")
    if not isinstance(y_val_targets_6_features, np.ndarray):
        raise TypeError(f"y_val_cvae_target (expected 6-feature array) is type {type(y_val_targets_6_features)}, expected np.ndarray.")

    if y_train_targets_6_features.shape[-1] != 6 or (len(y_train_targets_6_features.shape) != 2):
        raise ValueError(f"y_train_targets_6_features should be 2D with 6 features, but got shape {y_train_targets_6_features.shape}")
    if y_val_targets_6_features.shape[-1] != 6 or (len(y_val_targets_6_features.shape) != 2):
        raise ValueError(f"y_val_targets_6_features should be 2D with 6 features, but got shape {y_val_targets_6_features.shape}")
    
    # Ensure the number of samples match
    if x_train_data.shape[0] != y_train_targets_6_features.shape[0]:
        raise ValueError(f"Sample count mismatch between x_train_data ({x_train_data.shape[0]}) and y_train_targets_6_features ({y_train_targets_6_features.shape[0]})")
    if x_val_data.shape[0] != y_val_targets_6_features.shape[0]:
        raise ValueError(f"Sample count mismatch between x_val_data ({x_val_data.shape[0]}) and y_val_targets_6_features ({y_val_targets_6_features.shape[0]})")

    # --- Populate necessary dimensions in config ---
    
    # 1. window_size: Must be provided in the main configuration.
    if 'window_size' not in config or not isinstance(config.get('window_size'), int) or config.get('window_size') <= 0:
        # If x_train_data is already 3D from preprocessor, we can try to infer window_size
        if x_train_data is not None and len(x_train_data.shape) == 3:
            config['window_size'] = x_train_data.shape[1]
            print(f"Derived 'window_size' from 3D preprocessed training data: {config['window_size']}")
        else:
            raise ValueError(
                "'window_size' not found in config or is not a valid positive integer. "
                "It's required for the Conv1D encoder. Please add it to your main configuration file."
            )
    
    # 2. input_features_per_step: Number of features at each time step within the input window.
    if 'input_features_per_step' not in config or not isinstance(config.get('input_features_per_step'), int) or config.get('input_features_per_step') <= 0:
        if 'x_feature_dim' in config and isinstance(config['x_feature_dim'], int) and config['x_feature_dim'] > 0:
            # If old 'x_feature_dim' exists and likely means features per step
            config['input_features_per_step'] = config['x_feature_dim']
            print(f"Used existing 'x_feature_dim' for 'input_features_per_step': {config['input_features_per_step']}")
        elif x_train_data is not None:
            if len(x_train_data.shape) == 3: # Shape: (num_samples, window_size, features_per_step)
                # Ensure derived window_size matches if data is 3D
                if x_train_data.shape[1] != config['window_size']:
                    print(f"Warning: Mismatch between config 'window_size' ({config['window_size']}) and "
                          f"3D x_train_data.shape[1] ({x_train_data.shape[1]}). Using config 'window_size'.")
                config['input_features_per_step'] = x_train_data.shape[2]
                print(f"Derived 'input_features_per_step' from 3D training data: {config['input_features_per_step']}")
            elif len(x_train_data.shape) == 2: # Shape: (num_samples, features_per_step)
                config['input_features_per_step'] = x_train_data.shape[1]
                print(f"Derived 'input_features_per_step' from 2D training data: {config['input_features_per_step']}")
            else:
                raise ValueError(f"Cannot derive 'input_features_per_step' from x_train_data with shape {x_train_data.shape}")
        else:
            raise ValueError(
                "'input_features_per_step' could not be derived and is not set in config (nor as 'x_feature_dim'). "
                "It's required."
            )
    
    # The old 'x_feature_dim' specific check (lines 57-63 of your original snippet) can be removed
    # as 'input_features_per_step' now serves this purpose for the encoder.
    # If 'x_feature_dim' is used elsewhere, ensure it's consistent.

    # IMPORTANT: 'rnn_hidden_dim', 'conditioning_dim', and 'latent_dim' must be set in the config
    # and must be integers.
    
    required_dims = {
        'rnn_hidden_dim': "RNN hidden state dimension for CVAE components.",
        'conditioning_dim': "Conditioning vector dimension for CVAE components.",
        'latent_dim': "Latent space dimension for CVAE."
    }
    
    default_dims = { # Define sensible defaults if you prefer this over raising errors
        'rnn_hidden_dim': 64,
        'conditioning_dim': 10,
        'latent_dim': 32 
    }

    for dim_key, description in required_dims.items():
        if dim_key not in config or not isinstance(config.get(dim_key), int):
            # Option 1: Raise an error if not found or not an integer (safer)
            raise ValueError(f"'{dim_key}' not found in config or is not a valid integer. {description} It's required.")
            
            # Option 2: Use a default value (use with caution, ensure defaults match your model design)
            # config[dim_key] = default_dims[dim_key]
            # print(f"Warning: '{dim_key}' not found in config or not an integer. Using default: {config[dim_key]}. {description}")
        elif config.get(dim_key) <= 0: # Dimensions should be positive
             raise ValueError(f"'{dim_key}' in config must be a positive integer, but got {config.get(dim_key)}.")


    # --- Prepare h_context and conditions_t for training and validation ---
    # TODO: Replace this placeholder logic with your actual h_context and conditions_t generation.
    # These could come from datasets returned by preprocessor_plugin or be generated here.
    # Their dimensions must match config['rnn_hidden_dim'] and config['conditioning_dim'].
    
    num_train_samples = x_train_data.shape[0]
    h_context_train = datasets.get("h_train")
    if h_context_train is None:
        # Now config['rnn_hidden_dim'] is guaranteed to be an integer
        print(f"Warning: 'h_train' not found in datasets. Generating zeros for h_context_train (shape: ({num_train_samples}, {config['rnn_hidden_dim']})).")
        h_context_train = np.zeros((num_train_samples, config['rnn_hidden_dim']), dtype=np.float32)
    elif h_context_train.shape != (num_train_samples, config['rnn_hidden_dim']):
        raise ValueError(f"Shape mismatch for h_context_train. Expected ({num_train_samples}, {config['rnn_hidden_dim']}), got {h_context_train.shape}")

    conditions_t_train = datasets.get("cond_train")
    if conditions_t_train is None:
        # Now config['conditioning_dim'] is guaranteed to be an integer
        print(f"Warning: 'cond_train' not found in datasets. Generating zeros for conditions_t_train (shape: ({num_train_samples}, {config['conditioning_dim']})).")
        conditions_t_train = np.zeros((num_train_samples, config['conditioning_dim']), dtype=np.float32)
    elif conditions_t_train.shape != (num_train_samples, config['conditioning_dim']):
        raise ValueError(f"Shape mismatch for conditions_t_train. Expected ({num_train_samples}, {config['conditioning_dim']}), got {conditions_t_train.shape}")

    num_val_samples = x_val_data.shape[0]
    h_context_val = datasets.get("h_val")
    if h_context_val is None:
        # Now config['rnn_hidden_dim'] is guaranteed to be an integer
        print(f"Warning: 'h_val' not found in datasets. Generating zeros for h_context_val (shape: ({num_val_samples}, {config['rnn_hidden_dim']})).")
        h_context_val = np.zeros((num_val_samples, config['rnn_hidden_dim']), dtype=np.float32)
    elif h_context_val.shape != (num_val_samples, config['rnn_hidden_dim']):
        raise ValueError(f"Shape mismatch for h_context_val. Expected ({num_val_samples}, {config['rnn_hidden_dim']}), got {h_context_val.shape}")

    conditions_t_val = datasets.get("cond_val")
    if conditions_t_val is None:
        # Now config['conditioning_dim'] is guaranteed to be an integer
        print(f"Warning: 'cond_val' not found in datasets. Generating zeros for conditions_t_val (shape: ({num_val_samples}, {config['conditioning_dim']})).")
        conditions_t_val = np.zeros((num_val_samples, config['conditioning_dim']), dtype=np.float32)
    elif conditions_t_val.shape != (num_val_samples, config['conditioning_dim']):
        raise ValueError(f"Shape mismatch for conditions_t_val. Expected ({num_val_samples}, {config['conditioning_dim']}), got {conditions_t_val.shape}")


    cvae_train_inputs = [x_train_data, h_context_train, conditions_t_train]
    cvae_train_targets = y_train_targets_6_features # Use the 6-feature targets

    cvae_val_inputs = [x_val_data, h_context_val, conditions_t_val]
    cvae_val_targets = y_val_targets_6_features   # Use the 6-feature targets

    # Truncate validation inputs/targets if necessary (already handled for x_val_data by preprocessor or earlier logic)
    # This should be ensured by the preprocessor plugin or initial data loading.
    if cvae_val_inputs[0].shape[0] > cvae_train_inputs[0].shape[0]:
        print(f"[run_autoencoder_pipeline] Truncating validation CVAE inputs/targets to match training samples: {cvae_train_inputs[0].shape[0]}")
        for i in range(len(cvae_val_inputs)):
            cvae_val_inputs[i] = cvae_val_inputs[i][:cvae_train_inputs[0].shape[0]]
        cvae_val_targets = cvae_val_targets[:cvae_train_inputs[0].shape[0]]

    initial_latent_dim = config.get('initial_latent_dim', config['latent_dim']) # Use 'latent_dim' if 'initial_latent_dim' not set
    step_size_latent = config.get('step_size_latent', 8) # Step for adjusting latent_dim
    threshold_error = config.get('threshold_error', 0.1)
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 128)
    incremental_search = config.get('incremental_search', False) # Search for optimal latent_dim

    current_latent_dim = initial_latent_dim
    best_val_mae = float('inf')
    best_latent_dim = current_latent_dim
    best_autoencoder_manager = None

    # Loop for incremental search of latent_dim (optional)
    # If not using incremental_search, this loop will run once with config['latent_dim']
    while True:
        config['latent_dim'] = current_latent_dim # Set current latent_dim in config
        print(f"Training CVAE with latent_dim: {current_latent_dim}")
        
        autoencoder_manager = AutoencoderManager(encoder_plugin, decoder_plugin)
        
        # build_autoencoder now only takes config
        autoencoder_manager.build_autoencoder(config) 
        
        autoencoder_manager.train_autoencoder(
            data_inputs=cvae_train_inputs,
            data_targets=cvae_train_targets,
            epochs=epochs,
            batch_size=batch_size,
            config=config
        )
        
        # Evaluate
        # evaluate now returns a dictionary of metrics
        train_eval_results = autoencoder_manager.evaluate(cvae_train_inputs, cvae_train_targets, "Training", config)
        training_mae = train_eval_results.get('mae', float('nan')) # Get MAE from results
        print(f"Training MAE with latent_dim {current_latent_dim}: {training_mae}")
        print(f"Full Training Evaluation Results: {train_eval_results}")

        val_eval_results = autoencoder_manager.evaluate(cvae_val_inputs, cvae_val_targets, "Validation", config)
        validation_mae = val_eval_results.get('mae', float('nan'))
        print(f"Validation MAE with latent_dim {current_latent_dim}: {validation_mae}")
        print(f"Full Validation Evaluation Results: {val_eval_results}")
        
        if validation_mae < best_val_mae:
            best_val_mae = validation_mae
            best_latent_dim = current_latent_dim
            best_autoencoder_manager = autoencoder_manager # Save the manager with best model
            print(f"New best Validation MAE: {best_val_mae} with latent_dim: {best_latent_dim}")

        if not incremental_search:
            print("Incremental search disabled. Completing after one iteration.")
            break 
        
        # Incremental search logic (simplified from original)
        if validation_mae <= threshold_error:
            print(f"Threshold MAE ({threshold_error}) met or exceeded. Optimal latent_dim likely found: {current_latent_dim}")
            break
        else:
            if current_latent_dim >= config.get('max_latent_dim', 256): # Prevent excessive growth
                 print(f"Reached max_latent_dim ({config.get('max_latent_dim', 256)}). Stopping search.")
                 break
            current_latent_dim += step_size_latent
            if current_latent_dim <= 0: # Should not happen with positive step_size
                print("Latent dimension became non-positive. Stopping.")
                break
    
    if best_autoencoder_manager is None: # Should only happen if loop never ran or error
        if autoencoder_manager: # Fallback to last trained manager if search didn't improve
            best_autoencoder_manager = autoencoder_manager
            best_latent_dim = config['latent_dim'] # last used latent_dim
            print(f"Warning: Using last trained model as best. Latent_dim: {best_latent_dim}, Val MAE: {validation_mae}")
        else:
            raise RuntimeError("Autoencoder training failed or incremental search did not yield a model.")

    print(f"Final selected latent_dim: {best_latent_dim} with Validation MAE: {best_val_mae}")
    
    # Use the best manager for saving models
    encoder_model_filename = f"{config['save_encoder']}_ld{best_latent_dim}.keras"
    decoder_model_filename = f"{config['save_decoder']}_ld{best_latent_dim}.keras"
    best_autoencoder_manager.save_encoder(encoder_model_filename)
    best_autoencoder_manager.save_decoder(decoder_model_filename)
    print(f"Saved best encoder model to {encoder_model_filename}")
    print(f"Saved best decoder model to {decoder_model_filename}")

    end_time = time.time()
    execution_time = end_time - start_time
    
    # Fetch final MAE values from the best model's evaluation if possible, or last evaluation
    final_train_eval_results = best_autoencoder_manager.evaluate(cvae_train_inputs, cvae_train_targets, "Final Training", config)
    final_training_mae = final_train_eval_results.get('mae', float('nan'))
    final_val_eval_results = best_autoencoder_manager.evaluate(cvae_val_inputs, cvae_val_targets, "Final Validation", config)
    final_validation_mae = final_val_eval_results.get('mae', float('nan'))

    debug_info = {
        'execution_time_seconds': execution_time,
        'best_latent_dim': best_latent_dim,
        'encoder_plugin_params': encoder_plugin.get_debug_info(),
        'decoder_plugin_params': decoder_plugin.get_debug_info(),
        'final_validation_mae': final_validation_mae,
        'final_training_mae': final_training_mae,
        'final_validation_metrics': final_val_eval_results,
        'final_training_metrics': final_train_eval_results,
        'config_used': config # Save a snapshot of the config
    }

    from tensorflow.keras.utils import plot_model
    if 'save_log' in config and config['save_log']:
        save_debug_info(debug_info, config['save_log'])
        print(f"Debug info saved to {config['save_log']}.")
    
    if 'remote_log' in config and config['remote_log']:
        remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
        print(f"Debug info saved to {config['remote_log']}.")
    
    if 'model_plot_file' in config and config['model_plot_file'] and best_autoencoder_manager.model:
        try: 
            plot_filename = f"{config.get('model_plot_file', 'model_plot')}_ld{best_latent_dim}.png"
            plot_model(best_autoencoder_manager.model, to_file=plot_filename, show_shapes=True, show_layer_names=True, dpi=300)
            print(f"Model plot saved: {plot_filename}")
        except Exception as e: 
            print(f"WARN: Failed model plot: {e}")
        
    print(f"Pipeline execution time: {execution_time:.2f} seconds")


def load_and_evaluate_encoder(config):
    """
    Load and evaluate a pre-trained CVAE encoder component.
    """
    # Transformer specific loading (can be kept if you still use standalone transformer encoders)
    # For CVAE encoder component, custom_objects might not be needed unless plugin uses them.
    custom_objects = {}
    if config.get('encoder_plugin', '').lower() == 'transformer': # Example if transformer plugin has custom layers
        # Add transformer specific custom objects if needed by the plugin's model
        pass 
    
    encoder_model_path = config['load_encoder']
    encoder_component_model = load_model(encoder_model_path, custom_objects=custom_objects, compile=False)
    print(f"CVAE Encoder component model loaded from {encoder_model_path}")

    # Load input data (x_t values)
    # IMPORTANT: This data is assumed to be the x_t part of the CVAE input.
    x_t_data = load_csv(
        file_path=config['input_file'],
        headers=config.get('headers', False),
        force_date=config.get('force_date', False)
    ).to_numpy() # Convert to numpy array, assuming it's (num_samples, x_feature_dim)
    
    num_samples = x_t_data.shape[0]

    # TODO: Prepare h_context and conditions_t for encoder evaluation.
    # These must match the dimensions the loaded encoder component expects.
    # These dimensions should be available in the config used to train the model,
    # or inferred if the plugin's load method repopulates them.
    rnn_hidden_dim_eval = config.get('rnn_hidden_dim') # Get from config
    conditioning_dim_eval = config.get('conditioning_dim') # Get from config
    if rnn_hidden_dim_eval is None or conditioning_dim_eval is None:
        raise ValueError("rnn_hidden_dim and conditioning_dim must be in config for encoder evaluation.")

    print(f"Generating placeholder h_context and conditions_t for encoder evaluation.")
    h_context_eval = np.zeros((num_samples, rnn_hidden_dim_eval), dtype=np.float32)
    conditions_t_eval = np.zeros((num_samples, conditioning_dim_eval), dtype=np.float32)
    
    encoder_inputs_eval = [x_t_data, h_context_eval, conditions_t_eval]

    print(f"Encoding data. Input shapes: x_t: {encoder_inputs_eval[0].shape}, h_context: {encoder_inputs_eval[1].shape}, conditions_t: {encoder_inputs_eval[2].shape}")
    # The CVAE encoder component outputs [z_mean, z_log_var]
    encoded_outputs = encoder_component_model.predict(encoder_inputs_eval, verbose=1)
    z_mean_data = encoded_outputs[0]
    z_log_var_data = encoded_outputs[1] # z_log_var might also be useful to save/analyze
    
    print(f"Encoded z_mean shape: {z_mean_data.shape}, z_log_var shape: {z_log_var_data.shape}")

    # Save z_mean (or both)
    if config.get('evaluate_encoder'):
        output_filename = config['evaluate_encoder']
        # For simplicity, saving only z_mean. Adapt if you need to save z_log_var or a combined representation.
        encoded_df = pd.DataFrame(z_mean_data)
        
        # Add headers like 'latent_feature_0', 'latent_feature_1', ...
        encoded_df.columns = [f'latent_feature_{i}' for i in range(z_mean_data.shape[1])]
        
        # original_dates logic (if applicable and data aligns)
        # This part needs careful alignment if x_t_data was windowed or transformed.
        # Assuming x_t_data corresponds row-wise to original dates for now.
        original_data_for_dates = load_csv(file_path=config['input_file'], headers=config.get('headers', False), force_date=config.get('force_date', False))
        if config.get('force_date', False) and hasattr(original_data_for_dates, 'index') and original_data_for_dates.index.name is not None:
            if len(original_data_for_dates.index) == len(encoded_df):
                encoded_df.index = original_data_for_dates.index[:len(encoded_df)]
                encoded_df.index.name = 'date'
                write_csv(output_filename, encoded_df, include_date=True, headers=True, force_date=True)
            else:
                print("Warning: Date index length mismatch for encoded data. Saving without dates.")
                write_csv(output_filename, encoded_df, include_date=False, headers=True)
        else:
            write_csv(output_filename, encoded_df, include_date=False, headers=True)
        print(f"Encoded z_mean data saved to {output_filename}")


def load_and_evaluate_decoder(config):
    """
    Load and evaluate a pre-trained CVAE decoder component.
    """
    decoder_model_path = config['load_decoder']
    decoder_component_model = load_model(decoder_model_path, compile=False)
    print(f"CVAE Decoder component model loaded from {decoder_model_path}")

    # IMPORTANT: The 'input_file' for decoder evaluation should contain samples of the latent variable z_t.
    # It should NOT be the original raw data unless it has been processed into z_t samples.
    z_t_data = load_csv(
        file_path=config['input_file'], # This file should contain z_t samples
        headers=config.get('headers', False), # Adjust if z_t file has headers
        force_date=config.get('force_date', False) # Adjust if z_t file has dates
    ).to_numpy() # Assuming (num_samples, latent_dim)

    num_samples = z_t_data.shape[0]
    if z_t_data.shape[1] != config.get('latent_dim'):
         print(f"Warning: Loaded z_t data feature count ({z_t_data.shape[1]}) does not match config's latent_dim ({config.get('latent_dim')}). Ensure correct z_t input file.")
    
    # TODO: Prepare h_context and conditions_t for decoder evaluation.
    # These must match the dimensions the loaded decoder component expects.
    rnn_hidden_dim_eval = config.get('rnn_hidden_dim')
    conditioning_dim_eval = config.get('conditioning_dim')
    if rnn_hidden_dim_eval is None or conditioning_dim_eval is None:
        raise ValueError("rnn_hidden_dim and conditioning_dim must be in config for decoder evaluation.")

    print(f"Generating placeholder h_context and conditions_t for decoder evaluation.")
    h_context_eval = np.zeros((num_samples, rnn_hidden_dim_eval), dtype=np.float32)
    conditions_t_eval = np.zeros((num_samples, conditioning_dim_eval), dtype=np.float32)

    decoder_inputs_eval = [z_t_data, h_context_eval, conditions_t_eval]

    print(f"Decoding data. Input shapes: z_t: {decoder_inputs_eval[0].shape}, h_context: {decoder_inputs_eval[1].shape}, conditions_t: {decoder_inputs_eval[2].shape}")
    # The CVAE decoder component outputs the reconstructed x_prime_t
    decoded_data = decoder_component_model.predict(decoder_inputs_eval, verbose=1)
    print(f"Decoded data (reconstructed x_prime_t) shape: {decoded_data.shape}")

    # Save the decoded (reconstructed) data
    if config.get('evaluate_decoder'):
        output_filename = config['evaluate_decoder']
        decoded_df = pd.DataFrame(decoded_data)
        
        # Add headers like 'reconstructed_feature_0', 'reconstructed_feature_1', ...
        decoded_df.columns = [f'reconstructed_feature_{i}' for i in range(decoded_data.shape[1])]

        # Date handling: If the input z_t file had dates and they align, use them.
        # This is complex as z_t might not directly correspond to original dates.
        # For simplicity, saving without dates unless explicitly handled.
        # If original_input_for_dates is needed, load it separately.
        # original_data_for_dates = load_csv(file_path=config['original_data_for_dates_file'], ...) # If needed
        
        # Assuming for now that if force_date is true, the z_t input file might have dates.
        input_z_data_for_dates = load_csv(file_path=config['input_file'], headers=config.get('headers', False), force_date=config.get('force_date', False))
        if config.get('force_date', False) and hasattr(input_z_data_for_dates, 'index') and input_z_data_for_dates.index.name is not None:
            if len(input_z_data_for_dates.index) == len(decoded_df):
                decoded_df.index = input_z_data_for_dates.index[:len(decoded_df)]
                decoded_df.index.name = 'date'
                write_csv(output_filename, decoded_df, include_date=True, headers=True, force_date=True)
            else:
                print("Warning: Date index length mismatch for decoded data. Saving without dates.")
                write_csv(output_filename, decoded_df, include_date=False, headers=True)
        else:
            write_csv(output_filename, decoded_df, include_date=False, headers=True)
        print(f"Decoded data saved to {output_filename}")



