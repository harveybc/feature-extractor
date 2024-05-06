from setuptools import setup, find_packages

setup(
    name='feature-extractor',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'feature_extractor.encoders': [
            'default_encoder=app.encoder:DefaultEncoder',
            'rnn=app.plugins.encoder_plugin_rnn:RNNEncoderPlugin',
            'transformer=app.plugins.encoder_plugin_transformer:TransformerEncoderPlugin',
            'lstm=app.plugins.encoder_plugin_lstm:LSTMEncoderPlugin',
            'cnn=app.plugins.encoder_plugin_cnn:CNNEncoderPlugin'  # Updated plugin name and class
        ],
        'feature_extractor.decoders': [
            'default_decoder=app.decoder:DefaultDecoder',
            'rnn=app.plugins.decoder_plugin_rnn:RNNDecoderPlugin',
            'transformer=app.plugins.decoder_plugin_transformer:TransformerDecoderPlugin',
            'lstm=app.plugins.decoder_plugin_lstm:LSTMDecoderPlugin',
            'cnn=app.plugins.decoder_plugin_cnn:CNNDecoderPlugin'  # Updated plugin name and class
        ]
    },
    install_requires=[
        'keras',
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow'
    ],
    author='HArvey Bastidas',
    author_email='your.email@example.com',
    description='A feature extraction system that supports dynamic loading of encoder and decoder plugins for processing time series data.'
)
