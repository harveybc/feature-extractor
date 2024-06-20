import numpy as np
import pytest
from app.plugin_loader import load_encoder_decoder_plugins

def test_data_processing():
    encoder_plugin, _, decoder_plugin, _ = load_encoder_decoder_plugins('default', 'default')
    
    # Configure the plugin size
    encoder_plugin.configure_size(input_dim=128, encoding_dim=4)
    
    # Generate some dummy data
    data = np.random.rand(100, 128)
    
    # Train the encoder
    encoder_plugin.train(data)
    
    # Encode the data
    encoded_data = encoder_plugin.encode(data)
    
    # Ensure the encoded data has the expected shape
    assert encoded_data.shape == (100, 4)
