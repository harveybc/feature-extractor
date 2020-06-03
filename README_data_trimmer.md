# Preprocessor: Data-Trimmer

A simple data pre-processor that trims the constant valued columns.  Also removes rows from the start and the end of a dataset with features with consecutive zeroes. Usable both from command line and from class methods.

[![Build Status](https://travis-ci.org/harveybc/preprocessor.svg?branch=master)](https://travis-ci.org/harveybc/preprocessor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-preprocessor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/preprocessor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/preprocessor/blob/master/LICENSE)

## Description

Trims the constant valued columns. Also removes rows from the start and the end of a dataset with features with consecutive zeroes. 

The data-trimmer is implemented in the DataTrimmer class, it has methods for loading a dataset trimming it an producing an  output, please see [test_data_trimmer](https://github.com/harveybc/preprocessor/blob/master/tests/data_trimmer/test_data_trimmer.py), tests 1 to 3. It can also be used via command line, by default it performs auto-trimming, but it can be configured manually by using the --no_auto_trim option.

It also saves a configuration file, that is a CSV files with removed files and columns for applying similar  trimming to another dataset. Usable both from command line and from class methods (see [tests folder](https://github.com/harveybc/preprocessor/tree/master/tests)).

## Installation

The module is installed with the preprocessor package, the instructions are described in the [preprocessor README](../master/README.md).

### Command-Line Execution

The data-trimmer also is implemented as a console command:
> data-trimmer -- input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset to be trimmed.
* __--output_file <filename>__: (Optional) Filename for the output dataset. Defaults to the input dataset with the .output extension.
* __--output_config_file <filename>__: (Optional) Filename for the output configuration containing rows trimmed in columns 0 and columns trimmed in column 1. Defaults to the input dataset with the .config extension.
* __--input_config_file <filename>__: (Optional) Imports an existing configuration and trims a dataset with it.
* __--from_start <val>__:(Optional) number of rows to remove from the start of the input dataset.
* __--from_end <val>__: (Optional) number of rows to remove from the end of the input dataset.
* __--remove_columns__: (Optional) Removes all constant columns.
* __--no_auto_trim__: (Optional) Do not perform auto-trimming, useful if using the remove_columns, from_start or from_end options.

## Examples of usage
The following examples show both the class method and command line uses.

### Usage via Class Methods
```python
from preprocessor.data_trimmer.data_trimmer import DataTrimmer
# configure parameters (same vaiable names as command-line parameters)
class Conf:
    def __init__(self):
        self.input_file = "tests/data/test_input.csv"
conf = Conf()
# instance trimmer class and loads dataset
dt = DataTrimmer(conf)
# do the trimming
rows_t, cols_t = dt.trim_auto()
# save output to output file
dt.store()
```

### Usage via CLI

> data-trimmer --input_file "tests/data/test_input.csv"






