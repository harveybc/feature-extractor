# -*- coding: utf-8 -*-
"""
This File contains the LoadCSV class plugin. 
"""

from feature_extractor.plugin_base import PluginBase
from numpy import genfromtxt
from sys import exit

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class LoadCSV(PluginBase): 
    """ input plugin for the FeatureExtractor class, after initialization, the input_ds attribute is set """
    
    def __init__(self, conf):
        """ Initializes PluginBase. Do NOT delete the following line whether you have initialization code or not. """
        super().__init__(conf)
        # Insert your plugin initialization code here.
        pass

    def parse_cmd(self, parser):
        """ Adds command-line arguments to be parsed, overrides base class """
        parser.add_argument("--input_file", help="Input dataset file to load including path.", required=True)
        return parser
    
    def load_data(self):
        """ Load the input dataset """
        self.input_ds = genfromtxt(self.conf.input_file, delimiter=",")
        return self.input_ds
        
