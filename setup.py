from setuptools import setup, find_packages

setup(
    name='feature-extractor',
    version='1.0.0',
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='A feature extraction tool with support for dynamic plugin integration for encoders and decoders.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/harveybc/feature-extractor',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',  # Add your dependencies here
        'pandas',
        'tensorflow',  # Assuming TensorFlow is required for Keras
        'keras'
    ],
    entry_points={
        'feature_extractor.encoders': [
            'default_encoder = app.encoder:DefaultEncoder',
        ],
        'feature_extractor.decoders': [
            'default_decoder = app.decoder:DefaultDecoder',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.8',
)
