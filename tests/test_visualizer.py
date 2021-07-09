# -*- coding: utf-8 -*-

import pytest
import csv
import sys
import os
from filecmp import cmp
from data_logger import FeatureExtractor
import matplotlib.pyplot as plt
import requests

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class Conf:
    """ This method initialize the configuration variables for the visualization module  """
    
    def __init__(self):
        self.list_plugins = False
        self.config_file = os.path.join(os.path.dirname(__file__), "data/test_C05_config.JSON")
        
class TestMSSAPredictor:
    """ Component Tests """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()

    def test_C05T01_cmdline(self):
        """ Assess if a page can be downloaded and its size is bigger than the error page """
        # os.spawnl(os.P_DETACH, 'some_long_running_command')
        os.system("fe_data_logger --config_file "
            + self.conf.config_file
        )
        # assert if after the command is executed, a curl of the landing page returns code 200 and size > nbytes.
        response = requests.get('localhost:7777')
        print ("response.code", response.status_code)
        print ("response.content", response.content)
        print ("len(response.content)", len(response.content))
        assert (response.status == 200)
