
# Feature Extractor 

## Description

## Description

The Feature-extractor is a powerful and flexible tool designed for training autoencoders on CSV data, and also evaluating pre-trained encoders and decoders, with support for dynamic plugins. It utilizes various machine learning techniques to process time-series data through sliding windows and trains autoencoders with configurable parameters. The tool is capable of saving and loading models, making it highly versatile for both training and evaluation tasks.

### Key Features:
- **Dynamic Plugins:** Easily switch between different encoder and decoder plugins (e.g. ANN, CNN, LSTM, Transformer) to find the best fit for your data.
- **Configurable Parameters:** Customize the training process with parameters such as window size, initial size, step size, epochs, batch size, and error thresholds.
- **Model Management:** Save and load encoder and decoder models for reuse, avoiding the need to retrain models from scratch.
- **Remote Configuration:** Load and save configurations remotely, facilitating seamless integration with other systems and automation pipelines.
- **Incremental Search:** Optimize the interface size dynamically during training to achieve the best performance based on a specified error threshold.

This tool is designed for data scientists and machine learning engineers who need to preprocess and encode large datasets efficiently, and it can be easily integrated into larger machine learning workflows.


## Installation Instructions

To install and set up the feature-extractor application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/feature-extractor.git
    cd feature-extractor
    ```

2. **Create and Activate a Virtual Environment (Anaconda is required)**:

    - **Using `conda`**:
        ```bash
        conda create --name feature-extractor-env python=3.9
        conda activate feature-extractor-env
        ```

3. **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Build the Package**:
    ```bash
    python -m build
    ```

5. **Install the Package**:
    ```bash
    pip install .
    ```

6. **(Optional) Run the feature-extractor**:
    - On Windows, run the following command to verify installation (it generates an example output file csv_output.csv):
        ```bash
        feature-extractor.bat tests\data\csv_sel_unb_norm_512.csv 
        ```

    - On Linux, run:
        ```bash
        sh feature-extractor.sh tests\data\csv_sel_unb_norm_512.csv
        ```

7. **(Optional) Run Tests**:
For pasing remote tests, requires an instance of [harveybc/data-logger](https://github.com/harveybc/data-logger)
    - On Windows, run the following command to run the tests:
        ```bash
        set_env.bat
        pytest
        ```

    - On Linux, run:
        ```bash
        sh ./set_env.sh
        pytest
        ```

8. **(Optional) Generate Documentation**:
    - Run the following command to generate code documentation in HTML format in the docs directory:
        ```bash
        pdoc --html -o docs app
        ```
9. **(Optional) Install Nvidia CUDA GPU support**:

Please read: [Readme - CUDA](https://github.com/harveybc/feature-extractor/blob/master/README_CUDA.md)

## Usage

The application supports several command line arguments to control its behavior:

```
usage: feature-extractor.bat tests\data\csv_sel_unb_norm_512.csv
```

### Command Line Arguments

#### Required Arguments

- `input_file` (str): Path to the input CSV file.

#### Optional Arguments

- `-of, --output_file` (str): Path to the output CSV file.
- `-se, --save_encoder` (str): Filename to save the trained encoder model.
- `-sd, --save_decoder` (str): Filename to save the trained decoder model.
- `-le, --load_encoder` (str): Filename to load encoder parameters from.
- `-ld, --load_decoder` (str): Filename to load decoder parameters from.
- `-ee, --evaluate_encoder` (str): Filename for outputting encoder evaluation results.
- `-ed, --evaluate_decoder` (str): Filename for outputting decoder evaluation results.
- `-ep, --encoder_plugin` (str, default='default'): Name of the encoder plugin to use.
- `-dp, --decoder_plugin` (str, default='default'): Name of the decoder plugin to use.
- `-ws, --window_size` (int): Sliding window size to use for processing time series data.
- `-me, --threshold_error` (float): MSE error threshold to stop the training processes.
- `-is, --initial_size` (int): Initial size of the encoder/decoder interface.
- `-ss, --step_size` (int): Step size to adjust the size of the encoder/decoder interface.
- `-rl, --remote_log` (str): URL of a remote API endpoint for saving debug variables in JSON format.
- `-rlc, --remote_load_config` (str): URL of a remote JSON configuration file to download and execute.
- `-rsc, --remote_save_config` (str): URL of a remote API endpoint for saving configuration in JSON format.
- `-u, --username` (str): Username for the API endpoint.
- `-p, --password` (str): Password for the API endpoint.
- `-lc, --load_config` (str): Path to load a configuration file.
- `-sc, --save_config` (str): Path to save the current configuration.
- `-sl, --save_log` (str): Path to save the current debug info.
- `-qm, --quiet_mode` (flag): Suppress output messages.
- `-fd, --force_date` (flag): Include date in the output CSV files.
- `-inc, --incremental_search` (flag): Enable incremental search for interface size.
- `-hdr, --headers` (flag): Indicate if the CSV file has headers.

### Examples of Use

#### Autoencoder Training Example

To train an autoencoder using the CNN encoder and decoder plugins with a window size of 128, use the following command:

```bash
feature-extractor.bat tests\data\csv_sel_unb_norm_512.csv --encoder_plugin cnn --decoder_plugin cnn --window_size 128
```

#### Evaluating Data with a pre-trained Encoder
To evaluate data using a pre-trained encoder model, use the following command:

```bash
feature-extractor.bat tests\data\csv_sel_unb_norm_512.csv --load_encoder encoder_model.h5_col_0.keras
```

## Project Directory Structure
```md
feature-extractor/
│
├── app/                           # Main application package
│   ├── autoencoder_manager.py    # Manages autoencoder creation, training, saving, and loading
│   ├── cli.py                    # Handles command-line argument parsing
│   ├── config.py                 # Stores default configuration values
│   ├── config_handler.py         # Manages configuration loading, saving, and merging
│   ├── config_merger.py          # Merges configuration from various sources
│   ├── data_handler.py           # Handles data loading and saving
│   ├── data_processor.py         # Processes input data and runs the autoencoder pipeline
│   ├── main.py                   # Main entry point for the application
│   ├── plugin_loader.py          # Dynamically loads encoder and decoder plugins
│   ├── reconstruction.py         # Functionality for reconstructing data from autoencoder output
│   └── plugins/                       # Plugin directory
│       ├── decoder_plugin_ann.py         # Decoder plugin using an artificial neural network
│       ├── decoder_plugin_cnn.py         # Decoder plugin using a convolutional neural network
│       ├── decoder_plugin_lstm.py        # Decoder plugin using long short-term memory networks
│       ├── decoder_plugin_transformer.py # Decoder plugin using transformer layers
│       ├── encoder_plugin_ann.py         # Encoder plugin using an artificial neural network
│       ├── encoder_plugin_cnn.py         # Encoder plugin using a convolutional neural network
│       ├── encoder_plugin_lstm.py        # Encoder plugin using long short-term memory networks
│       └── encoder_plugin_transformer.py # Encoder plugin using transformer layers
│
├── tests/              # Test modules for the application
│   ├── acceptance          # User acceptance tests
│   ├── system              # System tests
│   ├── integration         # Integration tests
│   └── unit                # Unit tests
│
├── README.md                     # Overview and documentation for the project
├── requirements.txt              # Lists Python package dependencies
├── setup.py                      # Script for packaging and installing the project
├── set_env.bat                   # Batch script for environment setup
├── set_env.sh                    # Shell script for environment setup
└── .gitignore                         # Specifies intentionally untracked files to ignore
```

## Contributing

Contributions to the project are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to make contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

