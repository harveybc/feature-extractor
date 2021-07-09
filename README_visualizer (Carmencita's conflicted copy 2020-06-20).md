# FeatureExtractor: data_logger Component

Uses a Web UI to visualize plots and statistics with the data generated during feaure-extractor training or evaluation.

[![Build Status](https://travis-ci.org/harveybc/data_logger.svg?branch=master)](https://travis-ci.org/harveybc/data_logger)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-data_logger.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/data_logger?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/data_logger/blob/master/LICENSE)

## Description

Visualize via Web, data obtained from an input plugin, the data obtained via the input plugin may contain batch or real-time results of multiple feature-extractor trainers or evaluators (from now on called processes).  By default uses a Sqlite input plugin.  

It uses multiple output visualization plugins, each of which generate a new element in the feature-extractor dashboard views of the feature-extractor processes.  By default uses output plugins for: real-time MSE plot during training and batch calculated MSE plot from the evaluation of a pre-trained feature-extractor on a validation dataset. 

The data_logger uses a JSON configuration file for setting the Web service parameters and the configuration of the input and output plugins.

## Installation

The component is pre-installed with the data_logger package, the instructions are described in the [data_logger README](../master/README.md).

Additionally, environment variables must be set:

* For Linux and Mac:

> export FLASK_APP=data_logger/data_logger
> export FLASK_ENV=development

* For Windows:

> set FLASK_APP=data_logger\\data_logger
> set FLASK_ENV=development

### Command-Line Execution

For now, the execution is made vá the following commands (in the future a WSGI server line waitress will be used):

> flask run

### Configuration File

The data_logger uses a configuration file located in the data_logger/data_logger directory that sets the Web service parameters and the configuration of the input and output plugins.

The following is the default JSON configuration file:


# TODO: PEGAR CONFIG FILE CUANDO ESTÉ LISTO
{
  "service_port": "7777",
  "input_plugin": "vis_input_sqlite",
  "output_plugins": true,
  "age": 27,
  "address": {
    "streetAddress": "21 2nd Street",
    "city": "New York",
    "state": "NY",
    "postalCode": "10021-3100"
  },
  "phoneNumbers": [
    {
      "type": "home",
      "number": "212 555-1234"
    },
    {
      "type": "office",
      "number": "646 555-4567"
    }
  ],
  "children": [],
  "spouse": null
}






.






.