# -*- coding: utf-8 -*-

import pytest
import csv
import sys
import os
from filecmp import cmp
from data_logger import FeatureExtractor

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class Conf:
    """ This method initialize the configuration variables for a plugin """
    
    def __init__(self):
        """ Component Tests Constructor """
        fname = os.path.join(os.path.dirname(__file__), "data/test_input.csv")
        self.input_file = fname
        """ Test dataset filename """
        fname = os.path.join(os.path.dirname(__file__), "data/test_c02_output.csv")
        self.output_file = fname
        """ Output dataset filename """
        self.list_plugins = False
        self.core_plugin = "heuristic_ts"
        # in test data, column 10 is ema 10
        self.ema_fast = 0
        # in test data, column 24 is ema 20
        self.ema_slow = 1
        self.forward_ticks = 5  
        self.use_current = False  

class TestHeuristicTS:
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.rows_d, self.cols_d = self.get_size_csv(self.conf.input_file)
        """ Get the number of rows and columns of the test dataset """
        try:
            os.remove(self.conf.output_file)
        except:
            print("No test output file found.")
            pass

    def get_size_csv(self, csv_file):
        """ Get the number of rows and columns of a test dataset, used in all tests.

        Args:
        csv_file (string): Path and filename of a test dataset

        Returns:
        (int,int): number of rows, number of columns
        """
        rows = list(csv.reader(open(csv_file)))
        return (len(rows), len(rows[0]))


    def test_C02T01_core(self):
        """ Loads HeuristicTS using parameters from setup_method() and Asses that output file has 1 column and num_ticks - forward_ticks """
        self.fe = FeatureExtractor(self.conf)
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assertion
        assert (cols_o == 1) and (rows_o == self.fe.ep_core.rows_d - self.fe.ep_core.conf.forward_ticks)

    def test_C02T02_cmdline(self):
        """ same as C01T02, but via command-line """
        os.system("data_logger --core_plugin heuristic_ts --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --forward_ticks "
            + str(self.conf.forward_ticks)
        )
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the number of rows an colums is less than the input dataset and > 0
        assert (cols_o == 1) and (rows_o == rows_d - self.conf.forward_ticks)

    def test_C02T03_cmdline_current(self):
        """ same as C01T02, but via command-line """
        os.system("data_logger --core_plugin heuristic_ts --input_file "
            + self.conf.input_file
            + " --output_file "
            + os.path.join(os.path.dirname(__file__), "data/test_c02_t03_output.csv")
            + " --forward_ticks "
            + str(self.conf.forward_ticks)
            + " --current"
        )
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(os.path.join(os.path.dirname(__file__), "data/test_c02_t03_output.csv"))
        # assert if the number of rows an colums is less than the input dataset and > 0
        assert (cols_o == 1) and (rows_o == rows_d - self.conf.forward_ticks)

    def test_C02T04_cmdline_current_10k(self):
        """ generate 10k registers """
        os.system("data_logger --core_plugin heuristic_ts --input_file "
            + os.path.join(os.path.dirname(__file__), "data/test_input_10k.csv")
            + " --output_file "
            + os.path.join(os.path.dirname(__file__), "data/test_c02_t04_output.csv")   
            + " --forward_ticks "
            + str(self.conf.forward_ticks)
            # in test data, column 10 is ema 10
            + " --ema_fast 10"
            # in test data, column 24 is ema 20
            + " --ema_slow 24"
            + " --current"
        )        
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(os.path.join(os.path.dirname(__file__), "data/test_input_10k.csv"))
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(os.path.join(os.path.dirname(__file__), "data/test_c02_t04_output.csv")   )
        # assert if the number of rows an colums is less than the input dataset and > 0
        assert (cols_o == 1) and (rows_o == rows_d - self.conf.forward_ticks)