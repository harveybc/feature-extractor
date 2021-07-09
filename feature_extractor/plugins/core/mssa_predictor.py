# -*- coding: utf-8 -*-
"""
This File contains the MSSAPredictor class plugin.
"""

from data_logger.plugin_base import PluginBase
import numpy as np
import sys
from pymssa import MSSA
import copy
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class MSSAPredictor(PluginBase):
    """ Core plugin for the FeatureExtractor class, after initialization, saves the data and after calling the store_data method """

    def __init__(self, conf):
        """ Initializes PluginBase. Do NOT delete the following line whether you have initialization code or not. """
        super().__init__(conf)
        # Insert your plugin initialization code here.
        pass

    def parse_cmd(self, parser):
        """ Adds command-line arguments to be parsed, overrides base class """
        parser.add_argument("--num_components", help="Number of SSA components per input feature. Defaults to 0 = Autocalculated usign Singular Value Hard Thresholding (SVHT).", default=0, type=int)
        parser.add_argument("--window_size", help="Size of the data window that is half of the segments in which the dataset will be divided for analysis.", default=30, type=int)
        parser.add_argument("--forward_ticks", help="Number of ticks in the future to predict.", default=10, type=int)
        parser.add_argument("--plot_prefix", help="Exports plots of each grouped channel superposed to the input dataset. Defaults to None.", default=None, type=str)
        parser.add_argument("--show_error", help="Calculate the Mean Squared Error (MSE) between the prediction and the input future value. Defaults to False", action="store_true", default=False)
        return parser

#TODO: SLIDING WINDOW Y PREDICCION
    def core(self, input_ds):
        """ Performs sliding-window mssa_decomposition and prediction of each input feature. """
        # get the size of the input dataset, try if there are more than one column, else, assign number of columns as 1
        try:
            self.rows_d, self.cols_d = input_ds.shape
        except:
            (self.rows_d,) = input_ds.shape
            self.cols_d = 1
            input_ds = input_ds.reshape(self.rows_d, self.cols_d)
        if self.conf.window_size > self.rows_d // 5:
            print("The window_size must be at maximum 1/5th of the rows of the input dataset")
            sys.exit()
        # create an empty array with the estimated output shape
        self.output_ds = np.empty(shape=(self.rows_d-(self.conf.window_size), self.cols_d))
        
        # center the input_ds before fitting
        in_means = np.nanmean(input_ds, axis=0)
        input_ds = input_ds - in_means

        # calculate the output by performing MSSA on <segments> number of windows of data of size window_size
        segments = (self.rows_d - (2*self.conf.window_size + self.conf.forward_ticks))
        grouped_output = []
        for i in range(0, segments):
            #progress = i*100/segments
            #print("Segment: ",i,"/",segments, "     Progress: ", progress," %" )
            # verify if i+(2*self.conf.window_size) is the last observation
            first = i 
            if (i != segments-1):
                last = i + (2 * self.conf.window_size)
            else:
                last = self.rows_d
            # slice the input_ds dataset in 2*self.conf.window_size ticks segments
            s_data_w = input_ds[first : last,:]
            # center the data before fitting
            # only the first time, run svht, in following iterations, use the same n_components, without executing the svht algo
            if i == 0:
                # uses SVHT for selecting number of components if required from the conf parameters
                if self.conf.num_components == 0:
                    mssa = MSSA(n_components='svht', window_size=self.conf.window_size, verbose=False)
                    mssa.fit(s_data_w)
                    print("Automatically Selected Rank (number of components)= ",str(mssa.rank_))
                    rank = int(mssa.rank_)
                else:
                    rank = self.conf.num_components
                    mssa = MSSA(n_components=rank, window_size=self.conf.window_size, verbose=False)
                    mssa.fit(s_data_w)
            else:
                mssa = MSSA(n_components=rank, window_size=self.conf.window_size, verbose=False)
                mssa.fit(s_data_w)

            # TODO : Con las componentes, generar la predicción y luego los plots para cada feature del input_ds
            fc = mssa.forecast(self.conf.forward_ticks, timeseries_indices=None)        
            
            # extracts the required tick from prediction for each feature in fc_col
            fc_col = fc[:,self.conf.forward_ticks-1]
            (rows_o,) = fc_col.shape
            # transpose the predictions into a row 
            fc_row = fc_col.reshape(1,rows_o)
            # extract the row of components for all features into a single column
            comp_col = mssa.components_[:,(2 * self.conf.window_size) -1 , :].sum(axis=1)
            (rows_o,) = comp_col.shape
            # transpose the sum of channels per feature into a row
            comp_row = comp_col.reshape(1,rows_o)
            
            
            # concatenate otput array with the new predictions (5 tick fw) and the component sum (last tick in segment before prediction) in another array for plotting
            if i == 0:
                self.output_ds = fc_row
                denoised = comp_row                
            else:
                self.output_ds = np.concatenate((self.output_ds, fc_row), axis = 0)
                denoised = np.concatenate((denoised, comp_row), axis = 0)
            # TODO: calculate error per feature
        # calcluate shape of output_ds
        try:
            rows_o, cols_o = self.output_ds.shape
        except:
            (rows_o,) = self.output_ds.shape
            cols_o = 1
            self.output_ds = self.output_ds.reshape(rows_o, cols_o)

        # calculate error on the last half of the input dataset
        #r2 = r2_score(input_ds[(2 * self.conf.window_size) + self.conf.forward_ticks-1 : self.rows_d-self.conf.forward_ticks-1, feature], self.output_ds[:rows_o-self.conf.forward_ticks, feature])
        #r2 = r2_score(input_ds[(2 * self.conf.window_size) + self.conf.forward_ticks-1 + (self.rows_d//2): self.rows_d-self.conf.forward_ticks-1, 0], self.output_ds[(self.rows_d//2):rows_o-self.conf.forward_ticks, 0])
        r2 = r2_score(input_ds[(self.rows_d-self.conf.forward_ticks-1)-(self.rows_d//2): self.rows_d-self.conf.forward_ticks-1, 0], self.output_ds[(rows_o-self.conf.forward_ticks)-(self.rows_d//2) :rows_o-self.conf.forward_ticks, 0])
        mse = mean_squared_error(input_ds[(self.rows_d-self.conf.forward_ticks-1)-(self.rows_d//2): self.rows_d-self.conf.forward_ticks-1, 0], self.output_ds[(rows_o-self.conf.forward_ticks)-(self.rows_d//2) :rows_o-self.conf.forward_ticks, 0])
        mae = mean_absolute_error(input_ds[(self.rows_d-self.conf.forward_ticks-1)-(self.rows_d//2): self.rows_d-self.conf.forward_ticks-1, 0], self.output_ds[(rows_o-self.conf.forward_ticks)-(self.rows_d//2) :rows_o-self.conf.forward_ticks, 0])
        self.error = r2
        # plots th original data, predicted data and denoised data.
        if self.conf.plot_prefix != None:
            # Graficar matriz de correlaciones del primero y  agrupar aditivamente los mas correlated.
            # genera gráficas para cada componente con valores agrupados
            # for the 5th and the next components, save plots containing the original and cummulative timeseries for the first data column
            # TODO: QUITAR CUANDO DE HAGA PARA TODO SEGMENTO EN EL DATASET; NO SOLO EL PRIMERO
            # TODO : QUITAR: TEST de tamaño de grouped_components_ dictionary
            feature = 0
            for feature in range(self.cols_d):
                fig, ax = plt.subplots(figsize=(18, 7))
                ax.plot(self.output_ds[:rows_o-self.conf.forward_ticks, feature], lw=3, c='steelblue', alpha=0.8, label='predicted')
                ax.plot(denoised[self.conf.forward_ticks:, feature], lw=3, c='darkgoldenrod', alpha=0.6, label='denoised')
                ax.plot(input_ds[(2 * self.conf.window_size) + self.conf.forward_ticks-1 : self.rows_d-self.conf.forward_ticks-1, feature], lw=3, alpha=0.2, c='k', label='original') 
                ax.set_title('Forecast R2 = {:.3f}   MSE = {:.3f}   MAE = {:.3f}'.format(r2,mse,mae))
                ax.legend() 
                fig.savefig(self.conf.plot_prefix + str(feature) + '.png', dpi=600)

        # shows error
        if self.conf.show_error == True:
            for feature in range(self.cols_d):
                print("Feature = ", str(feature), "R2 score = ", str(r2))
        return self.output_ds