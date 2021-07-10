# -*- coding: utf-8 -*-
"""
This File contains the HTML output visualizator plugin. 
"""

from feature_extractor.plugin_base import PluginBase
from numpy import savetxt
from sys import exit
import os

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class VisHtml(PluginBase): 
    """ Output plugin for the FeatureExtractor class, after initialization, saves the data and after calling the store_data method """

    def __init__(self, conf):
        """ Constructor using same parameters as base class """
        super().__init__(conf)
        # Insert your plugin initialization code here.
        pass

    def parse_cmd(self, parser):
        """ Adds command-line arguments to be parsed, overrides base class """
        parser.add_argument("--output_file", help="Output file to store the processed data.", default="output.csv")
        return parser

    def template_path(self, output_ds):
        """ return this module's path """
        return os.path.dirname(__file__)
            
    