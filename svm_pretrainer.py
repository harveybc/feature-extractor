# svm_pretrainer: Trains a svm for each action, exports predicted results as csv,
#                 prints nmse and exports svm pre-trained models to be used in a 
#                 q-agent.

import sys
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from numpy import genfromtxt
from numpy import shape
import numpy as np
## \class QPretrainer
## \brief Trains a SVM with data generated with q-datagen and export predicted data and model data.
class QPretrainer():
    
    ## init method
    ## Loads the training and validation datasets
    def __init__(self, key):
        # Training set
        self.ts = []
        # Validation set
        self.vs = []
        # Number of features in dataset
        self.num_f = 0        
        # Number of training signals in dataset
        self.num_s = 0
        # number of folds for cross validation during grid search svm parameter tunning
        self.nfolds=5
        # First argument is the training dataset
        self.ts_f = sys.argv[1]
        # Second is validation dataset 
        self.vs_f = sys.argv[2]
        # Third argument is the prefix (including path) for the dcn pre-trained models 
        # for the actions, all modes are files with .model extention and the prefix is
        # concatenated with a number indicating the action:
        # 0 = Buy/CloseSell/nopCloseBuy
        # 1 = Sell/CloseBuy/nopCloseSell
        # 2 = No Open Buy
        # 3 = No Open Sell
        self.model_prefix = sys.argv[2]
        # output models prefix
        self.model_prefix = sys.argv[3]
        

    ## Load  training and validation datasets, initialize number of features and training signals
    def load_datasets(self):
        ts = genfromtxt(self.ts_f, delimiter=',')
        vs = genfromtxt(self.vs_f, delimiter=',')
        # split training and validation sets into features and training signal for regression
        self.num_f = ts.shape[1] - 4
        self.num_s = 4
    
    ## Train SVMs with the training dataset using cross-validation error estimation
    ## Returns best parameters
    def train_models(self):
        test=0
        #converts to nparray
        self.ts = np.array(self.ts)
        x = self.ts[...,0:self.num_f-1]
        y = self.ts[...,self.num_f]
        # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        grid_search = GridSearchCV(svm.SVR(kernel='rbf'), param_grid, cv=self.nfolds)
        grid_search.fit(x, y)
        return grid_search.best_params_

    
    ## Evaluate the trained models in the validation set to obtain the error
    def evaluate_validation(self):
        #print the parameters found by the gridsearch
        a= self.train_models()
        test=0       
        
    ## Export the trained models and the predicted validation set predictions, print statistics 
    def export_models(self):
        test=0
        
# main function 
if __name__ == '__main__':
    pt = QPretrainer(3)
    pt.load_datasets()
    pt.train_models()
    pt.evaluate_validation()