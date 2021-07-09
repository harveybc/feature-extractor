# FeatureExtractor: MSSA-Decomposer

Performs Multivariate Singular Spectrum Analysis (MSSA) decomposition of an input dataset.

[![Build Status](https://travis-ci.org/harveybc/data_logger.svg?branch=master)](https://travis-ci.org/harveybc/data_logger)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-data_logger.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/data_logger?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/data_logger/blob/master/LICENSE)

## Description

Performs MSSA decomposition of an input dataset, uses a configurable number of output channels, optionally 
grouping similar components.

## NOTE: 

It centers each feature by substracting the mean, please be careful using this data since the mean value is not stored anywhere. A better approach is to standardize the input dataset before using mssa_decomposer.

## Installation

The plugin is pre-installed with the data_logger package, the instructions are described in the [data_logger README](../master/README.md).

### Command-Line Execution

The plugin's core method can be executed by loading the plugin by instantiating a FeatureExtractor class with plugin-specific configuration parameters and also, it can be used from the console command data_logger, available after installing the [data_logger package](../master/README.md):
> data_logger --core_plugin mssa_decomposer --input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset for the default data_logger input plugin (load_csv).
* __--output_file <filename>__: (Optional) Filename for the output dataset for the default data_logger output plugin (store_csv). Defaults to output.csv
* __--num_components <val>__:(Optional) Number of SSA components per input feature. Defaults to 0 = Autocalculated usign Singular Value Hard Thresholding (SVHT).
* __--window_size <val>__: (Optional) Size of the window used for analysis. Dividies the dataset in segments of 2*window_size ticks. Defaults to 30.
* __--group_file <filename>__: (Optional) Filename for the JSON file containing manually set feature groups. Use --plot_correlation to export a w-correlation matrix plot. Defaults to None.
* __--w_prefix <filename_prefix>__: (Optional) Exports plots of the w-correlation matrix for grouped components for each feature. Defaults to None.
* __--plots_prefix <filename_prefix>__: (Optional) Exports plots of each grouped channel superposed to the input dataset. Defaults to None.


## Examples of usage

The following example show how to configure and execute the core plugin.

```python
from data_logger import FeatureExtractor
# configure parameters (same variable names as command-line parameters)
class Conf:
    def __init__(self):
        self.core_plugin = "mssa_decomposer"
        self.input_file = "tests/data/test_input.csv"
# initialize instance of the Conf configuration class
conf = Conf()
# initialize and execute the core plugin, loading the dataset with the default data_logger 
# input plugin (load_csv), and saving the results using the default output plugin (store_csv). 
fe = FeatureExtractor(conf)
```







.






.