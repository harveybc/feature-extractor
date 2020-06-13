# -*- coding: utf-8 -*-

import pytest
import csv
import sys
import os
from filecmp import cmp
from feature_eng.feature_eng import FeatureEng
import matplotlib.pyplot as plt

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class Conf:
    """ This method initialize the configuration variables for a plugin """
    
    def __init__(self):
        """ Component Tests Constructor """
        self.input_file = os.path.join(os.path.dirname(__file__), "data/test_c02_t03_output.csv")
        """ Test dataset filename """
        self.output_file = os.path.join(os.path.dirname(__file__), "data/test_c04_output.csv")
        """ Output dataset filename """
        self.list_plugins = False
        self.core_plugin = "mssa_predictor"
        self.num_components = 4
        self.window_size = 30
        self.plot_prefix = None
        self.forward_ticks = 1
        self.show_error = True

class TestMSSAPredictor:
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


    def atest_C04T01_core(self):
        """ Loads plugin from FeatureEng using parameters from setup_method() and Asses that output file has same number of columns but less rows  """
        self.fe = FeatureEng(self.conf)
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assertion
        assert (cols_o == self.cols_d) and (rows_o == self.rows_d-(2*(self.conf.window_size+self.conf.forward_ticks)))

    def atest_C04T02_cmdline(self):
        """ same as C04T01, but via command-line """
        os.system("feature_eng --core_plugin mssa_predictor --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --num_components "
            + str(self.conf.num_components)
        )
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assertion
        assert (cols_o == self.cols_d) and (rows_o == self.rows_d-(2*(self.conf.window_size+self.conf.forward_ticks)))

    def atest_C04T03_plot_prefix(self):
        """  """
        os.system("feature_eng --core_plugin mssa_predictor --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --num_components "
            + str(self.conf.num_components)
            + " --plot_prefix "
            + os.path.join(os.path.dirname(__file__), "plots/c04_")
        ) 
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if there are 3 groups per feature in the output dataset
        #TODO: ASSERT IF PLOT FILE EXISTS
        # assertion
        assert (cols_o == self.cols_d) and (rows_o == self.rows_d-(2*(self.conf.window_size+self.conf.forward_ticks)))
    
    def test_C04T04_svht_plot_prefix(self):
        """  """
        os.system("feature_eng --core_plugin mssa_predictor --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --num_components 0"
            + " --plot_prefix "
            + os.path.join(os.path.dirname(__file__), "plots/svht_c04_")
        ) 
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        #TODO: ASSERT IF PLOT FILE EXISTS
        # assertion
        assert (cols_o == self.cols_d) and (rows_o < self.rows_d)

    def test_C04T05_svht_variable_window_size(self):
        """ manual test for plotting the error (r2 coeff) for a variable window_size """
        error_list = []
        for window_size in range(6,9,1):
            # re-instance class with the new configuration 
            conf = Conf()
            # use the output of the test 5 of the heuristic_ts component as input since it has 10k rows = 10 times the maximum window size
            conf.input_file = os.path.join(os.path.dirname(__file__), "data/test_c02_t04_output_std.csv")
            # plot prefix to generate a plot per test iteration
            conf.plot_prefix =  os.path.join(os.path.dirname(__file__), "plots/6_9_c04t05_" + str(window_size) + "_")
            # use svht for auto selecting the number of components per window_size
            conf.num_components = 0
            # setup window_size configuration parameters
            conf.window_size = window_size
            # instance class, previous 1
            conf.forward_ticks = 1
            fe = FeatureEng(conf)
            # save the error for plotting
            err = fe.ep_core.error
            if err < -2: 
                err = -2 + ((err+2)/1000)
            
            error_list.append(err)
            del fe
            del conf
        # plots the error for each window size
        fig, ax = plt.subplots(figsize=(18, 7))
        ax.plot(range(6,9,1), error_list,  lw=3, c='steelblue', alpha=0.8, label='r2 score')
        ax.legend()
        fig.savefig(os.path.join(os.path.dirname(__file__), "plots/6_9_c04t05_variable_window_size.png"), dpi=600)
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(os.path.join(os.path.dirname(__file__), "data/test_c02_t04_output.csv"))
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if there are 3 groups per feature in the output dataset
        # TODO: ASSERT RIGHT NUMBER OF ROWS AND IF PLOT EXISTS
        assert (cols_o == cols_d) and (rows_o < rows_d)
