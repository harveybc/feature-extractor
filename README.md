# Feature Extractor 

Plug-in based feature extractor, includes modules for a configurable keras model trainer, evaluator and a training/evaluation visualizer with Web interface and serverless database. __Work In Progress, NOT USABLE YET__.

[![Build Status](https://travis-ci.org/harveybc/feature-extractor.svg?branch=master)](https://travis-ci.org/harveybc/feature-extractor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-feature-extractor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/feature-extractor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/feature-extractor/blob/master/LICENSE)

## Description

Implements modular components for feature extraction, it can be expanded by installing plugins for each module, there are 3 modules implemented:
* [Trainer](../master/README_trainer.md): Trains a machine learning model and saves the pre-trained model (feature-extractor).
* [Evaluator](../master/README_evaluator.md): Transforms an input dataset using a pre-trained model.
* [Visualizer](../master/README_visualizer.md): Uses a Web UI to visualize plots or statistics with the data generated during training or evaluation (i.e. some error measurement).

There are also three types of plugins:
* Input plugins: load the data to be processed.
* Operations plugins: perform feature extraction operations on loaded data: training or evaluation. 
* Output plugins: save the results of the feature extraction operations: the pre-trained model (feature-extractor) for the trainer module and an output dataset (feature-extracted data) for the evaluator.
* Visualization plugins: save data to be plotted by the visualizer module during training or evaluation.

It includes some pre-installed plugins (feature extractors):
* Keras hybrid 1D DeepConv/LSTM trainer and evaluator.
* Keras autoencoder trainer and encoder evaluator.
* TODO: Neuroevolved autoencoder (cuando esté re-implementado NEAT optimizer)
* TODO: Multi-level optimization autoencoder (cuando esté re-implementado singularity)

Usable both from command line and from class methods library.

## Installation

To install the package via PIP, use the following command:

> pip install -i https://test.pypi.org/simple/ feature-extractor

Also, the installation can be made by clonning the github repo and manually installing it as in the following instructions.

### Github Installation Steps
1. Clone the GithHub repo:   
> git clone https://github.com/harveybc/feature-extractor
2. Change to the repo folder:
> cd feature-extractor
3. Install requirements.
> pip install -r requirements.txt
4. Install python package (also installs the console command data-trimmer)
> python setup.py install
5. Add the repo folder to the environment variable PYTHONPATH
6. (Optional) Perform tests
> python setup.py test
7. (Optional) Generate Sphinx Documentation
> python setup.py docs

### Command-Line Execution

Each module is implemented as a console command:

* Trainer: 
> fe-trainer --core_plugin conv_lstm_trainer --input_file <input_dataset> <optional_parameters>
* Evaluator: 
> fe-evaluator --core_plugin conv_lstm_evaluator --input_file <input_dataset> --model <pretrained_model> <optional_parameters>
* Visualizer: 
> visualizer --core_plugin sqlite --input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

Parameters of the trainer and evaluator modules:

* __--list_plugins__: Shows a list of available plugins.
* __--core_plugin <core_plugin_name>__: Feature engineering core operations plugin to process an input dataset.
* __--input_plugin <input_lugin_name>__: Input dataset importing plugin. Defaults to csv_input.
* __--output_plugin <output_plugin_name>__: Output dataset exporting plugin. Defaults to csv_output.
* __--visualizer_plugin <visualizer_plugin_name>__: Output dataset exporting plugin. Defaults to csv_output.
* __--help, -h__: Shows help.

## Examples of usage

The following examples show both the class method and command line uses for the trainer module, for examples of other plugins, please see the specific module´s documentation.

### Example: Usage via CLI to list installed plugins

> fe-trainer --list_plugins

### Example: Usage via CLI to execute an installed plugin with its parameters

> fe-trainer --core_plugin conv_lstm_trainer  --input_file "tests/data/test_input.csv"

### Example: Usage via Class Methods conv_lstm_trainer plugin)

The following example show how to configure and execute the core plugin of the trainer.

```python
from feature_extractor.feature_extractor import FeatureExtractor
# configure parameters (same variable names as command-line parameters)
class Conf:
    def __init__(self):
        self.core_plugin = "conv_lstm_trainer"
        self.input_file = "tests/data/test_input.csv"
# initialize instance of the Conf configuration class
conf = Conf()
# initialize and execute the core plugin, loading the dataset with the default feature_extractor 
# input plugin (load_csv), and saving the results using the default output plugin (store_csv). 
fe = FeatureExtractor(conf)
```
