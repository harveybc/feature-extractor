
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

#### Required Arguments:
- `csv_file`: Path to the CSV file to process. This is a required positional argument for specifying the CSV file that the feature-extractor tool will process.

#### Optional Arguments:
- `-se`, `--save_encoder`: Filename to save the trained encoder model. Specify this argument to set the filename for saving the encoder's parameters after training.
- `-sd`, `--save_decoder`: Filename to save the trained decoder model. Specify this argument to set the filename for saving the decoder's parameters after training.
- `-le`, `--load_encoder`: Filename to load encoder parameters from. Use this option to specify the file from which the encoder parameters should be loaded.
- `-ld`, `--load_decoder`: Filename to load decoder parameters from. Use this option to specify the file from which the decoder parameters should be loaded.
- `-ee`, `--evaluate_encoder`: Filename for outputting encoder evaluation results. This option sets the output file for storing the results of the encoder evaluation.
- `-ed`, `--evaluate_decoder`: Filename for outputting decoder evaluation results. This option sets the output file for storing the results of the decoder evaluation.
- `-ep`, `--encoder_plugin`: Name of the encoder plugin to use. Defaults to 'default_encoder'. This argument allows users to specify which encoder plugin the tool should use.
- `-dp`, `--decoder_plugin`: Name of the decoder plugin to use. Defaults to 'default_decoder'. This argument allows users to specify which decoder plugin the tool should use.
- `-ws`, `--window_size`: Sliding window size to use for processing time series data. Defaults to 10. This option sets the window size for processing the data.
- `-me`, `--max_error`: Maximum MSE error to stop the training process. Specify this option to set a threshold for the maximum mean squared error at which training should be terminated.
- `-is`, `--initial_size`: Initial size of the encoder/decoder interface. This parameter sets the starting size for the interface between the encoder and decoder during training.
- `-ss`, `--step_size`: Step size to reduce the size of the encoder/decoder interface on each iteration. This parameter determines how much to decrease the interface size after each training iteration.
- `-rl`, `--remote_log`: URL of a remote data-logger API endpoint. Specify this option to set the endpoint for remote logging and monitoring of the training process.
- `-rc`, `--remote_config`: URL of a remote JSON configuration file to download and execute. Use this argument to specify a remote configuration that should be automatically downloaded and applied.
- `-qm`, `--quiet_mode`: Do not show results on the console. Defaults to 0 (disabled). Set this to 1 to enable quiet mode, which suppresses output to the console during processing.

### Examples of Use

**Train Encoder and Save Model**

To train an encoder using an RNN model on your data with a sliding window size of 10:

```bash
python -m app.main --encoder_plugin rnn --csv_file path/to/your/data.csv --window_size 10 --save_encoder rnn_encoder.model
```

## Project Directory Structure

feature-extractor/
│
├── app/                           # Main application package
│   ├── __init__.py                    # Initializes the Python package
│   ├── main.py                        # Entry point for the application
│   ├── config.py                      # Configuration settings for the app
│   ├── cli.py                         # Command line interface handling
│   ├── data_handler.py                # Module to handle data loading
│   ├── encoder.py                     # Default encoder logic
│   ├── decoder.py                     # Default decoder logic
│   └── plugins/                       # Plugin directory
│       ├── __init__.py                # Makes plugins a Python package
│       ├── encoder_plugin_rnn.py
│       ├── encoder_plugin_transformer.py
│       ├── encoder_plugin_lstm.py
│       ├── encoder_plugin_cnn.py
│       ├── decoder_plugin_rnn.py
│       ├── decoder_plugin_transformer.py
│       ├── decoder_plugin_lstm.py
│       ├── decoder_plugin_cnn.py
│
├── tests/                             # Test modules for your application
│   ├── __init__.py                    # Initializes the Python package for tests
│   ├── test_encoder.py                # Tests for encoder functionality
│   └── test_decoder.py                # Tests for decoder functionality
│
├── setup.py                           # Setup file for the package installation
├── README.md                          # Project description and instructions
├── requirements.txt                   # External packages needed
└── .gitignore                         # Specifies intentionally untracked files to ignore

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

