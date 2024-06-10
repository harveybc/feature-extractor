import pytest
from app.config_handler import merge_config
from app.config_defaults import DEFAULT_VALUES

def test_merge_config():
    # Default values
    defaults = DEFAULT_VALUES

    # Simulated loaded config from a file
    file_config = {
        'window_size': 256,
        'training_batch_size': 64,
    }

    # CLI arguments
    cli_args = {
        'epochs': 20,
        'encoder_plugin': 'cli_encoder',
    }

    # Unknown args
    unknown_args = {
        'remote_log_url': 'http://example.com/log',
    }

    # Expected result after merging
    expected_config = defaults.copy()
    expected_config.update(file_config)
    expected_config.update(cli_args)
    expected_config.update(unknown_args)

    # Perform the merge
    merged_config = merge_config(defaults, cli_args, unknown_args)

    # Assert that the merged config matches the expected config
    assert merged_config == expected_config

    # Additional checks for specific parameters
    assert merged_config['window_size'] == 256
    assert merged_config['training_batch_size'] == 64
    assert merged_config['epochs'] == 20
    assert merged_config['encoder_plugin'] == 'cli_encoder'
    assert merged_config['remote_log_url'] == 'http://example.com/log'
