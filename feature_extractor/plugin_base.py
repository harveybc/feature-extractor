# -*- coding: utf-8 -*-
"""
This File contains the PluginBase class. 
"""

from numpy import genfromtxt
from sys import exit
import argparse

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class PluginBase(): 
    """ Base class for FeatureExtractor plugins """

    def __init__(self, conf):
        """ Constructor using command line arguments in the conf.args attribute """
        if conf['args'] != None:
            parser = argparse.ArgumentParser(
                description="PluginBase: Base class for FeatureExtractor plugins."
            )
            parser = self.parse_cmd(parser)
            self.conf, self.unknown = parser.parse_known_args(conf['args'])
        else:
            self.conf = conf

    def parse_cmd(self, parser):
        """ Adds command-line arguments to parse, to be overriden by plugin-specific arguments """
        pass
    