# svm_pretrainer: Trains a svm for each action, exports predicted results as csv,
#                 prints nmse and exports svm pre-trained models to be used in a 
#                 q-agent.

import sys
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from numpy import genfromtxt
from numpy import shape
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from joblib import dump, load

## \class QPretrainer
## \brief Trains a SVM with data generated with q-datagen and export predicted data and model data.
class QPretrainer():    
    ## init method
    ## Loads the training and validation datasets
    def __init__(self):
        # Training set
        self.ts = []
        # Validation set
        self.vs = []
        # Number of features in dataset
        self.num_f = 0        
        # Number of training signals in dataset
        self.num_s = 4
        # number of folds for cross validation during grid search svm parameter tunning
        self.nfolds=3
        # First argument is the training dataset, last 25% of it is used as validation set
        self.ts_f = sys.argv[1]
        # Third argument is the prefix (including path) for the dcn pre-trained models 
        # for the actions, all modes are files with .model extention and the prefix is
        # concatenated with a number indicating the action:
        # 0 = TP
        # 1 = SL
        # 2 = dInv
        # 3 = direction (1: buy, -1: sell)
        self.num_ticks = 0
        self.model_prefix = sys.argv[2]
        # svm model
        self.svr_rbf = []

    ## Load  training and validation datasets, initialize number of features and training signals
    def load_datasets(self):
        self.ts_g = genfromtxt(self.ts_f, delimiter=',')
        # split training and validation sets into features and training signal for regression
        self.num_f = self.ts_g.shape[1] - self.num_s
        self.num_ticks = self.ts_g.shape[0]
        # split dataset into 75% training and 25% validation 
        self.ts_s = self.ts_g[1:(3*self.num_ticks)//4,:]
        self.ts = self.ts_s.copy()
        self.vs_s = self.ts_g[(3*self.num_ticks)//4 : self.num_ticks,:]
        self.vs = self.vs_s.copy() 
    ## Train SVMs with the training dataset using cross-validation error estimation
    ## Returns best parameters
    def train_model(self, signal):
        #converts to nparray
        self.ts = np.array(self.ts)
        self.x = self.ts[1:,0:self.num_f]
        #if signal == 0:
        #    print("Training set self.x = ",self.x)
        # TEST, remve 1 and replace by self.num_f
        self.y = self.ts[1:,self.num_f + signal]                  
        #print("Training action (", signal, ") self.y = ", self.y)
        # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        #Cs = [2e-4, 2e-2, 2e-1, 2e0, 2e1, 2e2, 2e4]
        #gammas = [2e-20, 2e-10, 2e0, 2e10]
        epsilons = [ 1,1.1,1.5,1.9]
        Cs = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2]
        gammas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        param_grid = {'C': Cs, 'epsilon':epsilons}
        grid_search = GridSearchCV(svm.SVR(gamma="auto"),param_grid, cv=self.nfolds)
        grid_search.fit(self.x, self.y)
        return grid_search.best_params_
    
    ## Evaluate the trained models in the validation set to obtain the error
    def evaluate_validation(self, params, signal):
        self.vs = np.array(self.vs)
        # TODO: NO ES TS SINO VS
        self.x_v = self.vs[1:,0:self.num_f]
        # TEST, remve 1 and replace by self.num_f
        self.y_v = self.vs[1:,self.num_f + signal]
        # create SVM model with RBF kernel with existing parameters
        self.svr_rbf = svm.SVR(gamma="auto", C=params["C"], epsilon=params["epsilon"])
        # Fit the SVM modelto the data and evaluate SVM model on validation x
        self.x = self.ts[1:,0:self.num_f]
        self.y = self.ts[1:,self.num_f + signal]
        if signal == 0:
            print("Validation set self.x_v = ",self.x_v)
        #TODO, NO ES PREDICT X SINO X_V
        y_rbf = self.svr_rbf.fit(self.x, self.y).predict(self.x_v)
        if signal == 0:
            print("Validation set y_rbf = ",y_rbf)
        # plot original and predicted data of the validation dataset
        lw = 2
        # TODO: NO ES TS SINO VS
        x_seq = list(range(0, self.vs.shape[0]-1))
        # 0 = Buy/CloseSell/nopCloseBuy
        print("x_seq.len = ", len(x_seq) , "y.len = " ,len(self.y_v) )
        fig=plt.figure()
        plt.plot(x_seq, self.y_v, color='darkorange', label='data')
        plt.plot(x_seq, y_rbf, color='navy', lw=lw, label='RBF model')
        plt.xlabel('data')
        plt.ylabel('target')
        plt.title('Signal ' + str(signal))
        plt.legend()
        fig.savefig('predict_' + str(signal) + '.png')
        if signal==3:
            plt.show()
        else:
            plt.show(block=False)
        return mean_squared_error(self.y_v, y_rbf)
 
    ## Export the trained models and the predicted validation set predictions, print statistics 
    def export_model(self, signal):
        dump(self.svr_rbf, self.model_prefix + str(signal)+'.svm') 
        
# main function 
if __name__ == '__main__':
    pt = QPretrainer()
    pt.load_datasets()
    for i in range(0,4):
        print('Training model '+str(i))
        params = pt.train_model(i)
        print('best_params_' + str(i) + ' = ',params)
        mse = pt.evaluate_validation(params,i)
        print('mean_squared_error on validation set:' + str(i) + ' = ' + str(mse))
        pt.export_model(i)
    
    