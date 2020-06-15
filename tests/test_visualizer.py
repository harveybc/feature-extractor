# -*- coding: utf-8 -*-

import pytest
import csv
import sys
import os
from filecmp import cmp
from feature_extractor.feature_extractor import FeatureExtractor
import matplotlib.pyplot as plt

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class Conf:
    """ This method initialize the configuration variables for the visualization module  """
    
    def __init__(self):
        self.list_plugins = False
        self.config_file = os.path.join(os.path.dirname(__file__), "data/test_C05_config.JSON")
        
class TestMSSAPredictor:
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()

    def test_C05T01_cmdline(self):
        """ Assess if a page can be downloaded and its size is bigger than the error page """
        os.system("fe_visualizer --config_file "
            + self.conf.config_file
        )
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assertion
        assert (cols_o == self.cols_d) and (rows_o == self.rows_d-(2*(self.conf.window_size+self.conf.forward_ticks)))
