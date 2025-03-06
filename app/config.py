DEFAULT_VALUES = {
    'input_file': './tests/data/phase_3_normalized_d1.csv',
    'validation_file': './tests/data/phase_3_normalized_d3.csv',
    'test_file': './tests/data/phase_3_normalized_d2.csv',
    'output_file': './phase_3_csv_output.csv',
    'save_encoder': './phase_3_encoder_model.h5',
    'save_decoder': './phase_3_decoder_model.h5',
    'load_encoder': None,
    'load_decoder': None,
    'evaluate_encoder': './phase_3_encoder_eval.csv',
    'evaluate_decoder': './phase_3_decoder_eval.csv',
    'encoder_plugin': 'cnn',
    'decoder_plugin': 'cnn',
    'use_sliding_windows': True,
    'window_size': 256, 
    'threshold_error': 0.5,
    'initial_size': 8,
    'initial_layer_size': 128,
    'intermediate_layers': 3,
    'layer_size_divisor': 2,
    'step_size': 1,
    'save_log': './phase_3_debug_out.json',
    'remote_log': None,
    'remote_load_config': None,
    'remote_save_config': None,
    'username': None,
    'password': None, 
    'load_config': None,
    'save_config': './phase_3_config_out.json',
    'quiet_mode': False,
    'force_date': True,
    'incremental_search': True, # if false performs decresing search instead
    'headers': True,
    'epochs': 2000,  # Add epochs here
    'batch_size': 32,  # Add batch_size here (best: 32)
    'learning_rate': 0.00001,  # Add learning_rate here
    'dataset_periodicity': '1h',  # Add dataset_periodicity here, can be 1m, 5m, 15m, 30m, 1h, 4h, daily
    'max_steps':20000, # max number of rows to read from input file
    'use_mmd':True,
    'mmd_sigma': 1.0,  # adjust as needed
    'statistical_loss_weight': 1.0,  # adjust as needed
    'use_pos_enc': False,
    'l2_reg': 1e-4,
    'early_monitor': 'val_loss',
    'early_patience': 16
}

