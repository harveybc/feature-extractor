
# Feature Extractor 

## Description

Feature Extractor is a Python application designed for processing CSV data through customizable encoding and decoding workflows. The application supports dynamic plugin integration, allowing users to extend its capabilities by adding custom encoder and decoder models. This feature makes it particularly suitable for tasks that require specialized data processing, such as machine learning model training and evaluation.

## Installation

Follow these steps to install and set up the application:

### Prerequisites
- Python 3.8 or newer
- pip (Python package installer)

### Setting Up a Virtual Environment
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
# Clone the repository
git clone https://github.com/harveybc/feature-extractor.git
cd yourproject

# Install dependencies
pip install -r requirements.txt

# Install the application (optional)
python setup.py install
```

## Usage

The application supports several command line arguments to control its behavior:

```
usage: python -m yourapp.main [-h] [-ds SAVE_ENCODER] [-dl LOAD_DECODER_PARAMS]
                              [-el LOAD_ENCODER_PARAMS] [-ee EVALUATE_ENCODER]
                              [-de EVALUATE_DECODER] [-em ENCODER_PLUGIN]
                              [-dm DECODER_PLUGIN]
                              csv_file
```

### Command Line Arguments
- `csv_file`: The path to the CSV file to process.
- `-ds, --save_encoder`: Filename to save the trained encoder.
- `-dl, --load_decoder_params`: Load decoder parameters from a file.
- `-el, --load_encoder_params`: Load encoder parameters from a file.
- `-ee, --evaluate_encoder`: Filename for outputting encoder evaluation results.
- `-de, --evaluate_decoder`: Filename for outputting decoder evaluation results.
- `-em, --encoder_plugin`: Specify the encoder plugin to use.
- `-dm, --decoder_plugin`: Specify the decoder plugin to use.
- `-me, --minimum_error`: Minimum MSE error to stop the training process and export the trained encoder.

### Examples of Use

**Load and Process a CSV File**

```bash
python -m app.main data/sample.csv
```

**Train Encoder and Save Model**

```bash
python -m app.main data/sample.csv -ds saved_models/encoder.model
```

**Evaluate Encoder and Output Results**

```bash
python -m app.main data/sample.csv -ee outputs/encoder_evaluation.txt
```

**Use a Custom Plugin for Encoding**

```bash
python -m app.main data/sample.csv -em custom_encoder_plugin
```

## Contributing

Contributions to the project are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to make contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

