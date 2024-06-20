import pytest
import json
from app.main import load_config, save_config

def test_load_config():
    config_path = 'tests/integration_tests/test_config.json'
    config_data = {
        'encoder_plugin': 'ann',
        'decoder_plugin': 'cnn',
        'training_batch_size': 128,
        'epochs': 20
    }
    
    # Save the config data to a file
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    # Load the config
    loaded_config = load_config(config_path)
    
    # Verify the loaded config
    assert loaded_config == config_data

def test_save_config():
    config_path = 'tests/integration_tests/test_save_config.json'
    config_data = {
        'encoder_plugin': 'rnn',
        'decoder_plugin': 'lstm',
        'training_batch_size': 64,
        'epochs': 15
    }
    
    # Save the config
    save_config(config_path, config_data)
    
    # Load the config to verify
    with open(config_path, 'r') as f:
        loaded_config = json.load(f)
    
    assert loaded_config == config_data
