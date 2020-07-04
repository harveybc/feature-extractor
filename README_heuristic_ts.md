# Feature Engineering: heuristic_ts plugin

Core plugin that generates an ideal training signal for trading based on a feature forwarded a configurable number of ticks.

TODO: MULTI-SYMBOL

[![Build Status](https://travis-ci.org/harveybc/feature_extractor.svg?branch=master)](https://travis-ci.org/harveybc/feature_extractor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-feature_extractor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/feature_extractor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/feature_extractor/blob/master/LICENSE)

## Description

Generates a training signal using Exponential Moving Average (EMA) fast, forwarded a configurable number of ticks minus EMA slow.  The input dataset must contain the two EMA columns.  The column index of the EMA fast and slow are configurable.

## Installation

The module is installed with the feature_extractor package, the instructions are described in the [feature_extractor README](../master/README.md).

### Command-Line Execution

The heuristic_ts core can be executed by loading the plugin from a class method and also can be used from a console command using feature_extractor:
> feature_extractor --core_plugin heuristic_ts --input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset for the default feature_extractor input plugin (load_csv).
* __--output_file <filename>__: (Optional) Filename for the output dataset for the default feature_extractor output plugin (store_csv). Defaults to _output.csv
* __--ema_fast <val>__:(Optional) column index of the EMA fast in the input dataset. Defaults to 0.
* __--ema_slow <val>__: (Optional) column index of the EMA slow in the input dataset. Defaults to 1.
* __--forward_ticks <val>__: (Optional) Number of forward ticks for EMA fast defaults 10.
* __--current__: (Optional) Do not use future data but only past data for the training signal calculus.

## Example of usage

The following example show how to configure and execute the core plugin.

```python
from feature_extractor.feature_extractor import FeatureExtractor
# configure parameters (same variable names as command-line parameters)
class Conf:
    def __init__(self):
        self.core_plugin = "heuristic_ts"
        self.input_file = "tests/data/test_input.csv"
# initialize instance of the Conf configuration class
conf = Conf()
# initialize and execute the core plugin, loading the dataset with the default feature_extractor 
# input plugin (load_csv), and saving the results using the default output plugin (store_csv). 
fe = FeatureExtractor(conf)
```







.