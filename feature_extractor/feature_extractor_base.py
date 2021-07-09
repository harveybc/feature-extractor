# -*- coding: utf-8 -*-
""" This File contains the FeatureExtractor class, has methods for listing and loading plugins and execute their entry point. """

import argparse
import sys
import logging
import numpy as np
import csv
import pkg_resources

# from data_logger import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)

class FeatureExtractorBase():
    """ Base class For FeatureExtractor. """
    
    def __init__(self, conf):
        """ Initializes FeatureExtractorBase with the configuration loaded from a JSON file. 
        Args:
        conf (JSON): plugin configuration loaded from configuration file.
        """
        self.conf = conf
        if conf != None:         
            if 'args' not in conf:
                self.conf['args'] = None
                self.setup_logging(logging.DEBUG) 
                _logger.info("Starting data_logger via class constructor...")
                # list available plugins
                if 'list_plugins' in conf:
                    if self.conf['list_plugins'] == True:
                        _logger.debug("Listing plugins.")
                        self.find_plugins()
                        _logger.debug("Printing plugins.")
                        self.print_plugins()
                # execute core operations
                else:
                    # sets default values for plugins
                    if 'input_plugin' not in conf: 
                        self.conf['input_plugin'] = "load_csv"  
                        _logger.debug("Warning: input plugin not found, using load_csv")
                    if 'output_plugin' not in conf: 
                        self.conf['output_plugin'] = "store_csv"
                        _logger.debug("Warning: input plugin not found, using store_csv")
                    if 'core_plugin' not in conf: 
                        self.conf['core_plugin'] = None
                    self.core()

    def parse_cmd(self, parser):
        """ Adds command-line arguments to parse """
        parser.add_argument("--version", action="version", version="data_logger")
        parser.add_argument("--list_plugins", help="lists all installed external and internal plugins", default=False)
        parser.add_argument("--core_plugin", help="Plugin to load ", default="heuristic_ts")
        parser.add_argument("--input_plugin", help="Input plugin to load ", default="load_csv")
        parser.add_argument("--output_plugin", help="Output plugin to load", default="store_csv")
        parser.add_argument("-v","--verbose",dest="loglevel",help="set loglevel to INFO",action="store_const",const=logging.INFO)
        parser.add_argument("-vv","--very_verbose",dest="loglevel",help="set loglevel to DEBUG",action="store_const",const=logging.DEBUG)
        return parser
    
    def core(self):
        """ Core data_logger operations. """
        _logger.debug("Finding Plugins.")
        self.find_plugins()
        _logger.debug("Loading plugins.")
        self.load_plugins() 
        if self.conf['core_plugin'] != None:
            _logger.debug("Loading input dataset from the input plugin.")
            self.input_ds = self.ep_input.load_data() 
            _logger.debug("Performing core operations from the  core plugin.")
            self.output_ds = self.ep_core.core(self.input_ds) 
            logger.debug("Executing the output plugin.")
            self.ep_output.store_data(self.output_ds) 
            _logger.info("data_logger finished.")
    
    def setup_logging(self, loglevel):
        """Setup basic logging.
        Args:
        loglevel (int): minimum loglevel for emitting messages
        """
        logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
        logging.basicConfig(
            level=loglevel,
            stream=sys.stdout,
            format=logformat,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    def parse_args(self, args):
        """Parse command line parameters.

        Args:
        args ([str]): command line parameters as list of strings

        Returns:
        :obj:`argparse.Namespace`: command line parameters namespace
        """
        parser = argparse.ArgumentParser(
            description="FeatureExtractor: Feature engineering operations."
        )
        parser = self.parse_cmd(parser)
        self.conf, self.unknown = parser.parse_known_args(args)
        # assign as arguments, the unknown arguments from the parser
        self.conf['args'] = self.unknown


        