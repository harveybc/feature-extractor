# FeatureExtractor: Visualizer

Uses a Web UI to visualize plots and statistics with the data generated during feaure-extractor training or evaluation.

[![Build Status](https://travis-ci.org/harveybc/feature_eng.svg?branch=master)](https://travis-ci.org/harveybc/feature_eng)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-feature_eng.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/feature_eng?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/feature_eng/blob/master/LICENSE)

## Description

Visualize via Web, data obtained from an input plugin, the data obtained via the input plugin may contain batch or real-time results of multiple feature-extractor trainers or evaluators (from now on called processes).  By default uses a Sqlite input plugin.  

It uses multiple output visualization plugins, each of which generate a new element in the feature-extractor dashboard views of the feature-extractor processes.  By default uses output plugins for: real-time MSE visualization during training and batch calculated MSE from the evaluation of a pre-trained feature-extractor on a validation dataset. 

The visualizer uses a JSON configuration file for setting the Web service parameters and the configuration of the input and output plugins.

## Installation

The plugin is pre-installed with the feature_eng package, the instructions are described in the [feature_eng README](../master/README.md).

### Command-Line Execution

The plugin's core method can be executed by loading the plugin by instantiating a FeatureExtractor class with plugin-specific configuration parameters and also, it can be used from the console command feature_eng, available after installing the [feature_eng package](../master/README.md):
> feature_eng --core_plugin mssa_predictor --input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset for the default feature_eng input plugin (load_csv).
* __--output_file <filename>__: (Optional) Filename for the output dataset for the default feature_eng output plugin (store_csv). Defaults to output.csv
* __--num_components <val>__:(Optional) Number of SSA components per input feature. Defaults to 0 = Autocalculated usign Singular Value Hard Thresholding (SVHT).
* __--forward_ticks <val>__:(Optional) Number of ticks in the future to predict. Defaults to 10.
* __--window_size <val>__: (Optional) Size of the window used for analysis. The segment in which the analysis is made is of size 2*window_size. Defaults to 30.
* __--plots_prefix <filename_prefix>__: (Optional) Exports a plot of the prediction superposed to the input dataset. Defaults to None.
* __--show_error__: (Optional) Calculate the Mean Squared Error (MSE) between the prediction and the input future value. Defaults to False.

## Examples of usage

The following example show how to configure and execute the core plugin.

```python
from feature_eng.feature_eng import FeatureExtractor
# configure parameters (same variable names as command-line parameters)
class Conf:
    def __init__(self):
        self.core_plugin = "mssa_predictor"
        self.input_file = "tests/data/test_input.csv"
# initialize instance of the Conf configuration class
conf = Conf()
# initialize and execute the core plugin, loading the dataset with the default feature_eng 
# input plugin (load_csv), and saving the results using the default output plugin (store_csv). 
fe = FeatureExtractor(conf)
```







.






.