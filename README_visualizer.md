# FeatureExtractor: Visualizer Component

Uses a Web UI to visualize plots and statistics with the data generated during feaure-extractor training or evaluation.

[![Build Status](https://travis-ci.org/harveybc/feature_extractor.svg?branch=master)](https://travis-ci.org/harveybc/feature_extractor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-feature_extractor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/feature_extractor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/feature_extractor/blob/master/LICENSE)

## Description

Visualize via Web, data obtained from an input plugin, the data obtained via the input plugin may contain batch or real-time results of multiple feature-extractor trainers or evaluators (from now on called processes).  By default uses a Sqlite input plugin.  

It uses multiple output visualization plugins, each of which generate a new element in the feature-extractor dashboard views of the feature-extractor processes.  By default uses output plugins for: real-time MSE plot during training and batch calculated MSE plot from the evaluation of a pre-trained feature-extractor on a validation dataset. 

The visualizer uses a JSON configuration file for setting the Web service parameters and the configuration of the input and output plugins.

## Installation

The component is pre-installed with the feature_extractor package, the instructions are described in the [feature_extractor README](../master/README.md).

### Command-Line Execution

The plugin's core method can be executed by loading the plugin by instantiating a FeatureExtractor class with plugin-specific configuration parameters and also, it can be used from the console command fe_visualizer, available after installing the [feature_extractor package](../master/README.md):

> fe_visualizer --config <JSON_configuration_file>

### Command-Line Parameters

* __--config <filename>__: The only mandatory parameter, is the filename for the input dataset for the default feature_extractor input plugin (load_csv).
* __--list_plugins__: Shows a list of isntalled visualization plugins.

## Examples of usage

The following example shows the contents of a basic configuration file for a visualizer.


# TODO: PEGAR CONFIG FILE CUANDO ESTÉ LISTO







.






.