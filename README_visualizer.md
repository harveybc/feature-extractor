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

Environment variables must be set:

* For Linux and Mac:

> export FLASK_APP=feature_extractor/visualizer

> export FLASK_ENV=development

* For Windows:

> set FLASK_APP=feature_extractor\\visualizer

> set FLASK_ENV=development

The execution is made via the following commands (in the future a WSGI server line waitress will be used):

> flask run

For ease of use, a script for setting the environment variables and executing the app is included, it must be executed in the root feature-extractor directory, where the scripts are located:

* For Linux and Mac:

> visualizer.sh

* For Windows:

> visualizer.bat

## Usage

The Web intreface can be accessed by default at:

[localhost:5000](localhost:5000)

The default port can be modified by setting the FLASK_RUN_PORT environment variable or bly using the --port argument to the flask run command.

### Configuration File

The visualizer uses a configuration file located in the feature_extractor/visualizer directory that sets the Web service parameters and the configuration of the input and output plugins.

The following is the default JSON configuration file:


```
{
    "input_plugin": "vis_input_sqlite",
    "input_plugin_config": {
        "filename": "test/db/plots.sqlite",
        "tables": [
            {
                "table_name": "training_progress",
                "fields": [
                    "mse",
                    "mae",
                    "r2"
                ]
            },
            {
                "table_name": "validation_stats",
                "fields": [
                    "mse",
                    "mae",
                    "r2"
                ]
            },
            {
                "table_name": "validation_plots",
                "fields": [
                    "original",
                    "predicted"
                ]
            }
        ]
    },
    "output_plugin": "vis_output",
    "output_plugin_config": {
        "dashboard": [
            {
                "table_name": "training_progress",
                "online": true,
                "delay": 3,
                "points": 800,
                "title": "Progress of Last Training Process",
                "fields": [
                    "mse"
                ]
            },
            {
                "table_name": "validation_plots",
                "title": "Validation Data Plot"
            },
            {
                "table_name": "validation_stats",
                "title": "List of Feature Extractor Stats on Validation Data"
            }
        ],
        "views": [
            {
                "table_name": "training_progress",
                "title": "Progress of Training Process",
                "online": true,
                "delay": 3,
                "points": 1600,
                "fields": [
                    "mse",
                    "mae",
                    "r2"
                ]
            },
            {
                "table_name": "validation_plots",
                "title": "Validation Data Plot",
                "online":  false,
                "fields": [
                    "original",
                    "predicted"
                ]
            },
            {
                "table_name": "validation_stats",
                "title": "Feature Extractor Stats on Validation Data",
                "online": false,
                "fields": [
                    "mse",
                    "mae",
                    "r2"
                ]
            }
        ]
    }
}
```




.






.