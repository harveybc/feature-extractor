DEFAULT_VALUES = {
    # --- File Paths ---
    "use_normalization_json": "examples/data/phase_3/phase_3_debug_out.json", # Kept phase_3 specific
    "x_train_file": "examples/data/phase_3/normalized_d1.csv",
    "y_train_file": "examples/data/phase_3/normalized_d1.csv", # Assumed y_train is same as x_train for autoencoder
    "x_validation_file": "examples/data/phase_3/normalized_d2.csv",
    "y_validation_file": "examples/data/phase_3/normalized_d2.csv", # Assumed y_val is same as x_val
    "x_test_file": "examples/data/phase_3/normalized_d3.csv",
    "y_test_file": "examples/data/phase_3/normalized_d3.csv",     # Assumed y_test is same as x_test

    "output_file": "examples/results/phase_3_2/phase_3_2_cnn_25200_1h_prediction.csv", # Kept more specific phase_3_2
    "results_file": "examples/results/phase_3_2/phase_3_2_cnn_25200_1h_results.csv",
    "loss_plot_file": "examples/results/phase_3_2/phase_3_2_cnn_25200_1h_loss_plot.png",
    "model_plot_file": "examples/results/phase_3_2/phase_3_2_cnn_1h_model_plot.png",
    "uncertainties_file": "examples/results/phase_3_2/phase_3_2_cnn_25200_1h_uncertanties.csv", # Renamed from "uncertainty_file" for consistency
    "predictions_plot_file": "examples/results/phase_3_2/phase_3_2_cnn_25200_1h_predictions_plot.png",
    "stl_plot_file": "examples/results/phase_3_2/phase_3_2_25200_1h_stl_decomposition_plot.png",
    "wavelet_plot_file": "examples/results/phase_3_2/phase_3_2_25200_1h_wavelet_decomposition_plot.png",
    "tapper_plot_file": "examples/results/phase_3_2/phase_3_2_25200_1h_multi_tapper_decomposition_plot.png",
    
    'save_encoder': 'examples/results/phase_3_2/phase_3_2_cnn_25200_1h_encoder_model.h5',
    'save_decoder': 'examples/results/phase_3_2/phase_3_2_cnn_25200_1h_decoder_model.h5',
    'load_encoder': None,
    'load_decoder': None,
    'evaluate_encoder': './encoder_eval.csv',
    'evaluate_decoder': './decoder_eval.csv',

    # --- Plugin Configuration ---
    "preprocessor_plugin": "stl_preprocessor",
    'encoder_plugin': 'cnn', # Default encoder plugin
    'decoder_plugin': 'cnn', # Default decoder plugin

    # --- Data and Model Structure ---
    "target_column": "CLOSE", # Primary target for some preprocessing steps, CVAE targets are separate
    'cvae_target_feature_names': [ # ADDED/UPDATED: Extended list of CVAE target features
        'OPEN', 'LOW', 'HIGH', 'vix_close', 'BC-BO', 'BH-BL', # Original 6
        'S&P500_Close',                                      # S&P500
        'CLOSE_15m_tick_1', 'CLOSE_15m_tick_2', 'CLOSE_15m_tick_3', 'CLOSE_15m_tick_4', # 15-min ticks
        'CLOSE_15m_tick_5', 'CLOSE_15m_tick_6', 'CLOSE_15m_tick_7', 'CLOSE_15m_tick_8',
        'CLOSE_30m_tick_1', 'CLOSE_30m_tick_2', 'CLOSE_30m_tick_3', 'CLOSE_30m_tick_4', # 30-min ticks
        'CLOSE_30m_tick_5', 'CLOSE_30m_tick_6', 'CLOSE_30m_tick_7', 'CLOSE_30m_tick_8'
    ],
    'input_offset': 0,
    'window_size': 288,  # Kept later definition
    'l2_reg': 1e-6,          # Kept later definition (smaller L2)
    'max_steps_train': 25200, # Kept later definition
    'max_steps_val': 6300,    # This was only defined once
    'max_steps_test': 6300,   # Kept later definition
    'batch_size': 128,        # Kept later definition
    "rnn_hidden_dim": 64,
    "conditioning_dim": 10,
    "latent_dim": 32,         # Default latent_dim, can be overridden by search
    "intermediate_layers": 2, # For plugins that use this
    "initial_layer_size": 128,# For plugins that use this
    "layer_size_divisor": 2,  # For plugins that use this
    "activation": "tanh",     # Default activation for some plugin layers

    # --- Training Parameters ---
    'iterations': 1,          # Typically 1 unless doing multiple runs with different seeds
    'epochs': 1000,           # Kept later definition
    'learning_rate': 1e-4,    # Kept later definition
    'early_patience': 40,     # Kept later definition
    "min_delta": 1e-7,        # Kept later definition for early stopping / reduceLR
    "start_from_epoch": 15,   # Kept later definition
    
    # --- KL Divergence Annealing (CVAE specific) ---
    "kl_beta_start": 0.0001,
    "kl_beta": 1.0,                 # Final value for KL beta (used as kl_beta_end by annealing)
    "kl_anneal_epochs": 100,        # Only one "kl_anneal_epochs" needed
    # "kl_weight": 1e-6, # This seems like an alternative to annealing, or a fixed weight. Removed for clarity with annealing.

    # --- Loss Function Weights & Parameters ---
    'huber_delta': 1.0,
    'mmd_weight': 0.01,       # Kept later definition
    'mmd_sigma': 1.0,         # Kept later definition
    'skew_weight': 0.001,
    'kurtosis_weight': 0.001,
    'cov_weight': 0.0,
    # "mmd_lambda": 1e-2, # This seems like a duplicate/alternative to mmd_weight. Removed.
    # "overfitting_penalty": 0.1, # Unclear how this is used, removed for now.
    # "penalty_close_lambda":0.0001, # Unclear, removed.
    # "penalty_far_lambda":0.0001,   # Unclear, removed.

    # --- Feature Engineering & Preprocessing ---
    'use_sliding_windows': False, # This is a general flag, preprocessor might handle windowing.
    "use_returns": True,          # Kept later definition
    "dataset_periodicity": '1h',
    "use_stl": True,
    "stl_period":24,
    "use_wavelets": True,
    "use_multi_tapper": True,
    "target_scaling_factor":1000, # If used by preprocessor for specific targets

    # --- Evaluation & Plotting ---
    "mc_samples":20,              # Kept later definition (fewer samples)
    "plotted_horizon": 6,
    "plot_color_predicted": "red", # Kept later definition
    "plot_color_true": "blue",
    "plot_color_uncertainty": "green",
    "plot_color_target": "orange", # Added from later section
    "uncertainty_color_alpha": 0.01,
    "plot_points": 48,             # Kept later definition

    # --- Strategy (if used, seems disabled by default) ---
    "use_strategy": False,
    "strategy_plugin_group": "heuristic_strategy.plugins",
    "strategy_plugin_name": "ls_pred_strategy",
    "strategy_1h_prediction": "examples/results/phase_1/phase_1_cnn_25200_1h_prediction.csv",
    "strategy_1h_uncertainty": "examples/results/phase_1/phase_1_cnn_25200_1h_uncertanties.csv",
    "strategy_base_dataset": "examples/data/phase_1/phase_1_base_d3.csv",
    "strategy_load_parameters": "examples/data/phase_1/strategy_parameters.json",
    
    # --- Incremental Search (for latent_dim or other hyperparameters) ---
    'incremental_search': True,
    'threshold_error': 0.5,       # For search loop termination
    # 'initial_size': 48, # Marked as NOT USED, removed.
    # "interface_size": 48, # Marked as not used, removed.
    # 'step_size': 2,       # Ambiguous, could be for latent_dim search or other. Define more clearly if needed.
    
    # --- Logging and Config Management ---
    'save_log': './debug_out.json',
    'remote_log': None,
    'remote_load_config': None,
    'remote_save_config': None,
    'username': None,
    'password': None, 
    'load_config': None,          # Path to load a full config.json from
    'save_config': './config_out.json', # Path to save the current config to
    'quiet_mode': False,          # Controls verbosity (later definition was False)
    'force_date': True,           # Kept later definition
    'headers': True,              # Kept later definition (True for CSVs with headers)
    "optimizer_output_file": "optimizer_output.json", # If using a separate optimizer script

    # --- Parameters for specific plugins or advanced features ---
    "use_daily": False, # Example: if some plugin behaves differently for daily data
    "predicted_horizons": [1,2,3,4,5,6], # If model predicts multiple steps ahead

    # --- Early Stopping and ReduceLROnPlateau (ensure these are used by callbacks) ---
    'early_stopping_monitor': 'val_loss', # Default monitor for early stopping
    # 'early_patience' is already defined above
    'early_stopping_restore_best_weights': True,
    'reduce_lr_monitor': 'val_loss',    # Default monitor for ReduceLROnPlateau
    'reduce_lr_patience': 15,           # Patience for ReduceLROnPlateau
    'reduce_lr_factor': 0.2,            # Factor for ReduceLROnPlateau
    'reduce_lr_min_lr': 1e-6,           # Min LR for ReduceLROnPlateau
}

# ... rest of your config.py file (load_config_from_file, get_config, etc.)

