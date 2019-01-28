# svm_pretrainer: Trains a svm for each action, exports predicted results as csv,
#                 prints nmse and exports svm pre-trained models to be used in a 
#                 q-agent.

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from numpy import genfromtxt
from numpy import shape
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D,Conv1D, MaxPooling2D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD, Adamax
from keras.utils import multi_gpu_model
import tensorflow as tf
from tensorflow.python.client import device_lib


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
        self.num_features = 0
        self.window_size = 30
        # Number of training signals in dataset
        self.num_s = 19
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
        # con lr=0.0002 e= TODO
        self.learning_rate = 0.0002
        # con epochs 400, ave3 con 0.6 featureselect y batch size=1024  e=0.418 
        # con epochs 800, ave3 con 0.6 featureselect y batch size=480  e= 0.361 
        # con epochs 800, ave3 con 0.6 featureselect y batch size=1024  e=0.0.519
        # con epochs 200, ave3 con 0.6 featureselect y batch size=256  ave3= 0.343
        # con epochs 400, ave3 con 0.6 featureselect y batch size=256  ave3= TODO 
        # number of validation tests to avarage during each training
        self.num_tests = 3

    def set_dcn_model(self, regression):

        # Deep Convolutional Neural Network for Regression
        model = Sequential()
        model.add(Dropout(0.4,input_shape=(self.num_features,self.window_size)))
        model.add(Conv1D(512, 3))
        model.add(Activation('sigmoid'))
        # Sin batch_normalization daba: 0.204
        # Con batch normalization: e=0.168
        model.add(BatchNormalization())
        # Con dropout = 0.1, e=0.168
        # con dropout = 0.2, e=0.121
        # con dropout = 0.4, e= 0.114
        model.add(Dropout(0.4))
        # mejor config so far: D0.4-512,D0.2-64,d0.1-32,16d64 error_vs=0.1 con 400 epochs y lr=0.0002
        # sin batchNormalization, eva = 0.107
        model.add(Conv1D(32, 3))
        model.add(Activation('sigmoid'))
        #model.add(BatchNormalization())

        # sin otra capa de 32, eva5 = 0.107
        #model.add(Conv1D(32, 3))
        #model.add(Activation('sigmoid'))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.1))
        
        # con capa de 16 da   ave= 104
        model.add(Conv1D(16, 3))
        model.add(Activation('sigmoid'))
        model.add(BatchNormalization())

        #model.add(MaxPooling1D(pool_size=2, strides=2))
        # second set of CONV => RELU => POOL
       # model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
       # con d=0.1 daba 0.11 con loss=0.08
       # con d=0.2 daba 0.22 con loss=0.06
        model.add(Dense(64, activation='sigmoid', kernel_initializer='glorot_uniform')) # valor óptimo:64 @400k
       # model.add(Activation ('sigmoid'))
        #model.add(BatchNormalization())

        # output layer
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        # check if output layers is for classification or regression
        if regression:
            model.add(Dense(1, activation = 'linear'))
        else:
            model.add(Dense(1, activation = 'sigmoid'))
        # multi-GPU support
        #model = to_multi_gpu(model)
        #self.reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=1e-4)
        # use SGD optimizer
        opt = Adamax(lr=self.learning_rate)
        #opt = SGD(lr=self.learning_rate, momentum=0.9)
        #paralell_model = multi_gpu_model(model, gpus=2)
        paralell_model = model
        if regression:
            paralell_model.compile(loss="mae", optimizer=opt, metrics=["mse"])
        else:
            paralell_model.compile(loss="binary_crossentropy", optimizer="adamax", metrics=["accuracy"])
        #model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
        return paralell_model 

    ## Load  training and validation datasets, initialize number of features and training signals
    def load_datasets(self):
        self.ts_g = genfromtxt(self.ts_f, delimiter=',', skip_header = 1)
        # split training and validation sets into features and training signal for regression
        self.num_f = self.ts_g.shape[1] - self.num_s
        self.num_features = self.num_f // self.window_size
        self.num_ticks = self.ts_g.shape[0]
        # split dataset into 75% training and 25% validation 
        self.ts_s = self.ts_g[1:(11*self.num_ticks)//12,:]
        self.ts = self.ts_s.copy()
        self.vs_s = self.ts_g[(11*self.num_ticks)//12 : self.num_ticks,:]
        self.vs = self.vs_s.copy() 
    
    ## Generate DCN  input matrix
    def dcn_input(self, data):
        #obs_matrix = np.array([np.array([0.0] * self.num_features)]*len(data), dtype=object)
        obs_matrix = []
        obs = np.array([np.array([0.0] * self.window_size)] * self.num_features)
        # for each observation
        data_p = np.array(data)
        for i, ob in enumerate(data):
            # for each feature, add an array of window_size elements
            for j in range(0,self.num_features):
                #print("obs=",obs)
                #print("data_p=",data_p[i, j * self.window_size : (j+1) * self.window_size])
                obs[j] = data_p[i, j * self.window_size : (j+1) * self.window_size]
                #obs[j] = ob[0]
            obs_matrix.append(obs.copy())
        return np.array(obs_matrix)
        
    ## Train SVMs with the training dataset using cross-validation error estimation
    ## Returns best parameters
    def train_model(self, signal):
        #converts to nparray
        self.ts = np.array(self.ts)
        self.x_pre = self.ts[1:,0:self.num_f]
        self.x = self.dcn_input(self.x_pre)
        self.y = self.ts[1:,self.num_f + signal]                  
        # TODO: Cambiar var svr_rbf por p_model
        # setup the DCN model
        self.svr_rbf = self.set_dcn_model(True)
        # train DCN model with the training data
        #best res so far: batch_size = 100   epochs=self.epochs
        #con batch size=64, epochs=200, lr=0.0002 daba:  loss=0.0283, e_vs=0.313  , cada epoca tardaba: 6s con 1ms/step
        #con batch size=512(64*8): , daba: loss=0.243 vs_e=0.251(0.241) cada epoca tardaba: 3s con 580us/step
        #con batch size=1024(128*8): , daba: loss=0.1787(0.251) vs_e=0.229 cada epoca tardaba: 3s con 540us/step
        #con batch size=2048(256*8): , daba: loss=0.27 vs_e=0.26 cada epoca tardaba: 3s con 540/step
        self.svr_rbf.fit(self.x, self.y, batch_size=1024, epochs=self.epochs, verbose=1)
        return self.svr_rbf 

    
    ## Evaluate the trained models in the validation set to obtain the error
    def evaluate_validation(self, params, signal):
        self.vs = np.array(self.vs)
        self.x_v_pre = self.vs[1:,0:self.num_f]
        self.x_v = self.dcn_input(self.x_v_pre)
        # TEST, remve 1 and replace by self.num_f
        self.y_v = self.vs[1:,self.num_f + signal].astype(int)
        if signal == 0:
            print("Validation set self.x_v = ",self.x_v)
        # predict the class of in the validation set
        y_rbf = self.svr_rbf.predict_classes(self.x_v)
        if signal == 0:
            print("Validation set y_rbf = ",y_rbf)
        # plot original and predicted data of the validation dataset
        lw = 2
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
        return mean_squared_error(self.y_v, y_rbf)
    
 
 ## Train SVMs with the training dataset using cross-validation error estimation
    ## Returns best parameters
    def train_model_c(self, signal):
        #converts to nparray
        self.ts = np.array(self.ts)
        self.x_pre = self.ts[1:,0:self.num_f]
        self.x = self.dcn_input(self.x_pre)
        self.y = self.ts[1:,self.num_f + signal].astype(int)                  
        # TODO: Cambiar var svr_rbf por p_model
        # setup the DCN model
        self.svr_rbf = self.set_dcn_model(False)
        # train DCN model with the training data
        #best res so far: batch_size = 100   epochs=self.epochs
        #con batch size=64, epochs=200, lr=0.0002 daba:  loss=0.0283, e_vs=0.313  , cada epoca tardaba: 6s con 1ms/step
        #con batch size=512(64*8): , daba: loss=0.243 vs_e=0.251(0.241) cada epoca tardaba: 3s con 580us/step
        #con batch size=1024(128*8): , daba: loss=0.1787(0.251) vs_e=0.229 cada epoca tardaba: 3s con 540us/step
        #con batch size=2048(256*8): , daba: loss=0.27 vs_e=0.26 cada epoca tardaba: 3s con 540/step
        self.svr_rbf.fit(self.x, self.y, batch_size=256, epochs=self.epochs, verbose=1)
        return self.svr_rbf 

        
    ## Evaluate the trained models in the validation set to obtain the error
    def evaluate_validation_c(self, model, signal):
        self.vs = np.array(self.vs)
        self.x_v_pre = self.vs[1:,0:self.num_f]
        self.x_v = self.dcn_input(self.x_v_pre)
        # TEST, remve 1 and replace by self.num_f
        self.y_v = self.vs[1:,self.num_f + signal].astype(int)
        if signal == 0:
            print("Validation set self.x_v = ",self.x_v)
        # predict the class of in the validation set
        y_rbf = self.svr_rbf.predict_classes(self.x_v)
        if signal == 0:
            print("Validation set y_rbf = ",y_rbf)
        # plot original and predicted data of the validation dataset
        lw = 2
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
        if signal==16:
            plt.show()
        else:
            plt.show(block=False)
        return mean_squared_error(self.y_v, y_rbf)
 
    ## Export the trained models and the predicted validation set predictions, print statistics 
    def export_model(self, signal):
        self.svr_rbf.save(self.model_prefix + str(signal)+'.dcn') 
    
        
# main function 
if __name__ == '__main__':
    #print(device_lib.list_local_devices())
    print("TRAINING")
    pt = QPretrainer()
    pt.load_datasets()
    error = pt.num_s*[0.0]
    error_ant = pt.num_s*[0.0]
    error_accum = pt.num_s*[0.0]
    
    for j in range(0,pt.num_tests):
        print('test: ',j+1,'/',pt.num_tests)
        #for i in range(0,pt.num_s):
        for i in range(16,17):
            print('Training model '+str(i))
            error_ant[i] = error[i]
            # verifies if the actions are for classification(the last 6 ones)
            if (i>=10):
                params = pt.train_model_c(i)
                print('best_params_' + str(i) + ' = ',params)
                error[i] = pt.evaluate_validation_c(params,i)
                print('error on validation set:' + str(i) + ' = ' + str(error[i]))
            else:    
                params = pt.train_model(i)
                print('best_params_' + str(i) + ' = ',params)
                error[i] = pt.evaluate_validation(params,i)
                print('mean_squared_error on validation set:' + str(i) + ' = ' + str(error[i]))
            error_accum[i] += error[i]
            if j == pt.num_tests-1:
                print('average validation error:', error_accum[i]/pt.num_tests)
            if error[i] <= error_ant[i]:    
                pt.export_model(i)
    for i in range(16,19):
        print('Error in signal ', i, ": ", error_accum[i]/pt.num_tests )
       
    