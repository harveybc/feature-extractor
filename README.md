
# Feature Extractor 

## Description

Feature Extractor is a Python application designed for processing CSV data through customizable encoding and decoding workflows. The application supports dynamic plugin integration, allowing users to extend its capabilities by adding custom encoder and decoder models. 

This feature makes it particularly suitable for tasks that require specialized data processing, such as machine learning model training and evaluation. It Includes plugins for RNN, CNN, LSTM, and Transformer-based architectures.

## Installation

Follow these steps to install and set up the application:

### Prerequisites
- Python 3.8 or newer
- pip (Python package installer)

### Setting Up a Virtual Environment (optional)
It's recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

### Install the Application
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-github/feature-extractor.git
cd feature-extractor
pip install -r requirements.txt
python setup.py install
```

## Usage

The application supports several command line arguments to control its behavior:

```
usage: python -m app.main [-h] [-ds SAVE_ENCODER] [-dl LOAD_DECODER_PARAMS]
                              [-el LOAD_ENCODER_PARAMS] [-ee EVALUATE_ENCODER]
                              [-de EVALUATE_DECODER] [-em ENCODER_PLUGIN]
                              [-dm DECODER_PLUGIN]
                              csv_file
```

### Command Line Arguments
- `csv_file`: The path to the CSV file to process.
- `--encoder_plugin <name>`: Selects the encoder plugin. Available options include `rnn`, `cnn`, `lstm`, and `transformer`.
- `--decoder_plugin <name>`: Selects the decoder plugin. Corresponds to the encoders listed above.
- `--save_encoder <filename>`: Specifies the filename to save the trained encoder model.
- `--load_decoder_params <filename>`: Loads decoder parameters from a file.
- `--evaluate_encoder <filename>`: Outputs encoder evaluation results to a file.
- `--evaluate_decoder <filename>`: Outputs decoder evaluation results to a file.
- `--window_size <size>`: Defines the size of the sliding window used for processing the time series data.
- `--max_error <error>`: Sets the maximum mean squared error threshold for stopping the training.
- `--initial_size <size>`: Initial size of the encoder output/input of the decoder.
- `--step_size <size>`: Step size for reducing the size of the encoder/decoder interface.


### Examples of Use

**Train Encoder and Save Model**

To train an encoder using an RNN model on your data with a sliding window size of 10:

```bash
python -m app.main --encoder_plugin rnn --csv_file path/to/your/data.csv --window_size 10 --save_encoder rnn_encoder.model
```
### File Descriptions

- app/main.py: This is the main entry script where the application logic is handled based on command line arguments. It decides whether to train, evaluate the encoder, or evaluate the decoder based on input flags.

- app/config.py: Contains configuration settings, like paths and parameters that might be used throughout the application.

- app/cli.py: Handles parsing and validation of command line arguments using libraries such as argparse.

- app/data_handler.py: Responsible for loading and potentially preprocessing the CSV data.

- app/encoder.py and decoder.py: These files contain the default implementation of the encoder and decoder using Keras. They define simple artificial neural networks as starting points.

- app/plugins/init.py: Makes the plugins folder a package that can dynamically load plugins.

- app/plugins/encoder_plugin_cnn.py and decoder_plugin_cnn.py: Example plugins demonstrating how third-party plugins can be structured.

- tests/: Contains unit tests for the encoder, decoder, and other components of the application to ensure reliability and correctness.

- setup.py: Script for setting up the project installation, including entry points for plugin detection.

- README.md: Provides an overview of the project, installation instructions, and usage examples.

- requirements.txt: Lists dependencies required by the project which can be installed via pip.

- .gitignore: Lists files and directories that should be ignored by Git, such as __pycache__, environment-specific files, etc.


## Contributing

Contributions to the project are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to make contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

