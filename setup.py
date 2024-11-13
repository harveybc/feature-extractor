from setuptools import setup, find_packages

setup(
    name='feature-extractor',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'feature_extractor=app.main:main'
        ],
        'feature_extractor.encoders': [
            'default=app.plugins.encoder_plugin_ann:Plugin',
            'ann=app.plugins.encoder_plugin_ann:Plugin',
            'rnn=app.plugins.encoder_plugin_rnn:Plugin',
            'transformer=app.plugins.encoder_plugin_transformer:Plugin',
            'lstm=app.plugins.encoder_plugin_lstm:Plugin',
            'cnn=app.plugins.encoder_plugin_cnn:Plugin',
            'cnn_signed=app.plugins.encoder_plugin_cnn_signed:Plugin'
        ],
        'feature_extractor.decoders': [
            'default=app.plugins.decoder_plugin_ann:Plugin',
            'ann=app.plugins.decoder_plugin_ann:Plugin',
            'rnn=app.plugins.decoder_plugin_rnn:Plugin',
            'transformer=app.plugins.decoder_plugin_transformer:Plugin',
            'lstm=app.plugins.decoder_plugin_lstm:Plugin',
            'cnn=app.plugins.decoder_plugin_cnn:Plugin',
            'cnn_signed=app.plugins.decoder_plugin_cnn_signed:Plugin'
        ]
    },
    install_requires=[
            'tensorflow-gpu',
            'nvidia-cudnn-cu11',
            'numpy',
            'pandas',
            'h5py',
            'scipy',
            'build',
            'pytest',
            'pdoc3',
            'keras-multi-head',
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='A feature extraction system that supports dynamic loading of encoder and decoder plugins for processing time series data.'
)
