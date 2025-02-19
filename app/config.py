DEFAULT_VALUES = {
    'input_file': './tests/data/phase_1_normalized_d1.csv',
    'validation_file': './tests/data/phase_1_normalized_d2.csv',
    'output_file': './csv_output.csv',
    'save_encoder': './encoder_model.h5',
    'save_decoder': './decoder_model.h5',
    'load_encoder': None,
    'load_decoder': None,
    'evaluate_encoder': './encoder_eval.csv',
    'evaluate_decoder': './decoder_eval.csv',
    'encoder_plugin': 'transformer',
    'decoder_plugin': 'transformer',
    'use_sliding_windows': False,
    'window_size': 256, 
    'threshold_error': 0.5,
    'initial_size': 16,
    'step_size': 1,
    'future_shift': 1,
    'save_log': './debug_out.json',
    'remote_log': None,
    'remote_load_config': None,
    'remote_save_config': None,
    'username': None,
    'password': None, 
    'load_config': None,
    'save_config': './config_out.json',
    'quiet_mode': False,
    'force_date': True,
    'incremental_search': True, # if false performs decresing search instead
    'headers': True,
    'epochs': 600,  # Add epochs here
    'batch_size': 128,  # Add batch_size here
    'learning_rate': 0.0001,  # Add learning_rate here
    'dataset_periodicity': '1h'  # Add dataset_periodicity here, can be 1m, 5m, 15m, 30m, 1h, 4h, daily
}

