# svm_pretrainer: Trains a dcn for each action, exports predicted results as csv,
#                 prints nmse and exports svm pre-trained models to be used in a 
#                 q-agent.
# v2 uses both classification and regression signals and decide using both of them


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
from sklearn.metrics import mean_squared_error, accuracy_score
from joblib import dump, load
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D,Conv1D, MaxPooling2D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, TimeDistributed
from keras.optimizers import SGD, Adamax
from keras.utils import multi_gpu_model
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.layers import LSTM


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
        # Best so far 0.0001 error = 0.106 en 200 epochs, 2nd best, 0.0002 en 400 epochs=0.104
        # 0.002 (Adamax default) = 0.137
        # 0.0002 = 0.127
        # 0.0005 = 0.142
        # mejor leaning rate sin batch normalization + 0.0002
        self.learning_rate = 0.0008
        #epocsh 400, ava3 = TODO
        #epocsh 1200, ava3 = 0.66, loss=0.169
        self.epochs = 50 
        # number of validation tests to avarage during each training
        self.num_tests = 1

    def set_dcn_model_r_alex(self):
        # Deep Convolutional Neural Network for Regression
        model = Sequential()
        # for observation[19][48], 19 vectors of 128-dimensional vectors,input_shape = (19, 48)
        # model.add(Dropout(0.6,input_shape=(self.num_features,self.window_size)))
        model.add(Conv1D(48, 5, strides=2,use_bias=False, input_shape=(self.num_features,self.window_size)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(MaxPooling1D(pool_size=3, strides=2))
        
        model.add(Conv1D(128, 3, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
       
        model.add(MaxPooling1D(pool_size=3, strides=2))
        
        model.add(Conv1D(200, 3, stride=1, padding=1, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
       
        model.add(Conv1D(200, 3, stride=1, padding=1, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
       
        model.add(Conv1D(128, 3, stride=1, padding=1, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
       
        model.add(MaxPooling1D(pool_size=3, strides=2))
        
        model.add(Dense(640)) 
        model.add(Dropout(0.2))
        model.add(Dense(640)) 
        model.add(Dropout(0.2))
        model.add(Dense(256)) 
        model.add(Dropout(0.2))
        #model.add(Dense(32)) 
        #model.add(Dropout(0.4))
        model.add(Dense(1, activation = 'linear')) 

        # use SGD optimizer
        opt = Adamax(lr=self.learning_rate)

        #paralell_model = multi_gpu_model(model, gpus=2)
        paralell_model = model

        model.compile(loss="mae", optimizer=opt, metrics=["mse"])
        return paralell_model 

    def set_dcn_model_r(self):
        # Deep Convolutional Neural Network for Regression
        model = Sequential()
        # for observation[19][48], 19 vectors of 128-dimensional vectors,input_shape = (19, 48)
        # model.add(Dropout(0.6,input_shape=(self.num_features,self.window_size)))
        model.add(Conv1D(512, 5, strides=2,use_bias=False, input_shape=(self.num_features,self.window_size))) 
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        
        model.add(Conv1D(128, 3, use_bias=False))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
                
        #model.add(Dropout(0.6))
        #model.add(Conv1D(8, 3, use_bias=False))
        #model.add(BatchNormalization())
        #model.add(Activation('relu'))
        
        model.add(LSTM(units = 256, input_shape=(self.num_features,self.window_size))) 
        model.add(BatchNormalization()) 

        #model.add(LSTM(units = 32, return_sequences = True, dropout = 0.4,  input_shape=(self.num_features,self.window_size)))            
        #model.add(LSTM(units = 16, return_sequences = True, dropout = 0.4, input_shape=(self.num_features,self.window_size)))                        
        #model.add(LSTM(units=32, dropout = 0.4, recurrent_dropout = 0.6 ))
        #model.add(BatchNormalization())
 
        model.add(Dense(640)) 
        model.add(BatchNormalization())
        model.add(Activation('hard_sigmoid'))
        #model.add(Dropout(0.2))
        model.add(Dense(320)) 
        model.add(BatchNormalization())
        model.add(Activation('hard_sigmoid'))
        #model.add(Dropout(0.2))
        model.add(Dense(1, activation = 'linear')) 

        # use SGD optimizer
        opt = Adamax(lr=self.learning_rate)

        #paralell_model = multi_gpu_model(model, gpus=2)
        paralell_model = model 

        model.compile(loss="mae", optimizer=opt, metrics=["mse"])
        return paralell_model 

    def set_dcn_model_c(self):

        # Deep Convolutional Neural Network for Regression
        model = Sequential()
        # for observation[19][48], 19 vectors of 128-dimensional vectors,input_shape = (19, 48)
        # first set of CONV => RELU => POOL
        # mejor result 0.1 con dropout de 0.4 en 400 epochs con learning rate 0.0002 en config  521,64,32,16, en h4 2018 con indicator_period=70
        # 0.2,0.1,lr=0.0002 1200 eva: 0.117
        # 0.4,eva = 0.108
        model.add(Dropout(0.2,input_shape=(self.num_features,self.window_size)))
        model.add(Conv1D(512, 3))
        model.add(Activation('sigmoid'))
        # Sin batch_normalization daba: 0.204
        # Con batch normalization: e=0.168
        model.add(BatchNormalization())
        # Con dropout = 0.1, e=0.168
        # con dropout = 0.2, e=0.121
        # con dropout = 0.4, e= 0.114
        model.add(Dropout(0.2))
        #sin capa de LSTM50,  e=0.107
        #con capa de LSTM50, e= 0.191
        #model.add(LSTM(units = 50, return_sequences = True))
        
        #model.add(Dropout(0.2))
        # mejor config so far: D0.4-512,D0.2-64,d0.1-32,16d64 error_vs=0.1 con 400 epochs y lr=0.0002
        # sin capa de 64, eva = 0.114
        # on capa de 128, eva = 0.125
        # on capa de 32,  eva = 0.107
        # on capa de 16,  eva = 0.114
        model.add(Conv1D(32, 3))
        model.add(Activation('sigmoid'))
        #model.add(BatchNormalization())

        # con otra capa de 32, eva5 = 0.126
        # sin otra capa de 32, eva5 = 0.107, sin minmax normalization
        # sin otra capa de 32, eva5 = 0.124 , con minmax normalization antes de power transform
        #model.add(Conv1D(32, 3))
        #model.add(Activation('sigmoid'))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.1))
        
        # con capa de 16 da   eva5= 107
        model.add(Conv1D(16, 3))
        model.add(Activation('sigmoid'))
        model.add(BatchNormalization())

        #sin capa de LSTM50, eva3=0.104 probar con 400 epochs
        #con capa de LSTM50, eva3= 0.212
        #model.add(LSTM(units = 50, return_sequences = True))
        
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
        model.add(Dense(1, activation = 'sigmoid'))
        # multi-GPU support
        #model = to_multi_gpu(model)
        #self.reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=1e-4)
        # use SGD optimizer
        opt = Adamax(lr=self.learning_rate)
        #opt = SGD(lr=self.learning_rate, momentum=0.9)
        #paralell_model = multi_gpu_model(model, gpus=2)
        paralell_model = model
        paralell_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        #model.compile(loss="binary_crossentropy", optimizer="adamax", metrics=["accuracy"])
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
        self.ts_s = self.ts_g[0:(3*self.num_ticks)//4,:]
        self.ts = self.ts_s.copy()
        
        #TODO: TEST: QUITAR hasta print
        #ts_n = np.array(self.ts)
        #y_t = ts_n[0:,self.num_f + 0]         
        #print("y_t = ", y_t[0:30] )
        
        self.vs_s = self.ts_g[(3*self.num_ticks)//4 : self.num_ticks,:]
        self.vs = self.vs_s.copy() 
        
        #TODO: TEST: QUITAR hasta print
        #vs_n = np.array(self.vs)
        #y_v = vs_n[0:,self.num_f + 0]         
        #print("y_v = ", y_v)
    
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
    ## Returns best parameterse
    def train_model(self, signal):
        #converts to nparray
        # TODO: Usando dataset completo ts_g en lugar de solo ts,incluyendo validation set, se hace separación por parámetro validation_split de fit
        self.ts = np.array(self.ts_g)
        self.x_pre = self.ts[0:,0:self.num_f]
        
        #TODO: BBORRAR hasta print
        #print("self.x_pre[0:30, self.num_f-1] = ", self.x_pre[0:30, self.num_f-1])
        
        self.x = self.dcn_input(self.x_pre)
        self.y = self.ts[0:,self.num_f + signal]         
        
        #print("signal = ",signal,"   self.y = ", self.y)         
        # TODO: Cambiar var svr_rbf por p_model
        
        # setup the DCN model
        self.svr_rbf = self.set_dcn_model_r()
        # train DCN model with the training data
        #best res so far: batch_size = 100   epochs=self.epochs
        #con batch size=64, epochs=200, lr=0.0002 daba:  loss=0.0283, e_vs=0.313  , cada epoca tardaba: 6s con 1ms/step
        #con batch size=512(64*8): , daba: loss=0.243 vs_e=0.251(0.241) cada epoca tardaba: 3s con 580us/step
        #con batch size=1024(128*8): , daba: loss=0.1787(0.251) vs_e=0.229 cada epoca tardaba: 3s con 540us/step
        #con batch size=2048(256*8): , daba: loss=0.27 vs_e=0.26 cada epoca tardaba: 3s con 540/step
        history = self.svr_rbf.fit(self.x, self.y, validation_split=0.25, batch_size=1024, epochs=self.epochs, verbose=1)
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        fig = plt.figure()
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.title('model mse')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig.savefig('predict_' + str(signal) + '_mse.png')
        # summarize history for loss
        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig.savefig('predict_' + str(signal) + '_loss.png')
        
        return self.svr_rbf 

    
    ## Evaluate the trained models in the validation set to obtain the error
    def evaluate_validation(self, params, signal):
        self.vs = np.array(self.vs)
        self.x_v_pre = self.vs[0:,0:self.num_f]
        self.x_v = self.dcn_input(self.x_v_pre)
        
        # TEST, remve 1 and replace by self.num_f
        self.y_v = self.vs[0:,self.num_f + signal]
        #print("signal = ",signal,"   self.y_v = ", self.y_v)
        #if signal == 0:
        #    print("Validation set self.x_v = ",self.x_v)
        # predict the class of in the validation set
        y_rbf = self.svr_rbf.predict(self.x_v)
        #if signal == 0:
        #    print("Validation set y_rbf = ",y_rbf)
        # plot original and predicted data of the validation dataset
        lw = 2
        x_seq = list(range(0, self.vs.shape[0])) 
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
    
 
 
 # TODO: COREGIR RANGOS DE TS Y VS PARA TRAIN_MODEL_C Y EVALUATE_C
  ## Returns best parameters
    def train_model_c(self, signal):
        #converts to nparray
        self.ts = np.array(self.ts)
        self.x_pre = self.ts[1:,0:self.num_f]
        self.x = self.dcn_input(self.x_pre)
        self.y = self.ts[1:,self.num_f + signal].astype(int)                  
        # TODO: Cambiar var svr_rbf por p_model
        # setup the DCN model
        self.svr_rbf = self.set_dcn_model_c()
        # train DCN model with the training data 
        # con batch size =  128 dio 0.17
        # con batch_size = 1024, ev=0.75(0.1)
        # con batch_size = 512, ev = 0.176
        self.svr_rbf.fit(self.x, self.y, batch_size=1024, epochs=self.epochs, verbose=1)
        return self.svr_rbf 
        
    ## Evaluate the trained models in the validation set to obtain the error
    def evaluate_validation_c(self, model, signal):
        self.vs = np.array(self.vs)
        self.x_v_pre = self.vs[1:,0:self.num_f]
        self.x_v = self.dcn_input(self.x_v_pre)
        # TEST, remve 1 and replace by self.num_f
        self.y_v = self.vs[1:,self.num_f + signal].astype(int)
        #if signal == 0:
        #    print("Validation set self.x_v = ",self.x_v)
        # predict the class of in the validation set
        y_rbf = self.svr_rbf.predict_classes(self.x_v)
        #if signal == 0:
        #    print("Validation set y_rbf = ",y_rbf)
        # plot original and predicted data of the validation dataset
        lw = 2
        # TODO: NO ES TS SINO VS
        x_seq = list(range(0, self.vs.shape[0]-1))
        # 0 = Buy/CloseSell/nopCloseBuy
        #rint("x_seq.len = ", len(x_seq) , "y.len = " ,len(self.y_v) )
        #fig=plt.figure()
        #plt.plot(x_seq, self.y_v, color='darkorange', label='data')
        #plt.plot(x_seq, y_rbf, color='navy', lw=lw, label='RBF model')
        #plt.xlabel('data')
        #plt.ylabel('target')
        #plt.title('Signal ' + str(signal))
        #plt.legend()
        #fig.savefig('predict_cpu_' + str(signal) + '.png')
        #if signal==18:
        #    plt.show()
        #else:
        #    plt.show(block=False)
        return accuracy_score(self.y_v, y_rbf)
 
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
    # for i in range(10, 11):
    for i in range(8,9):
        print('Training model '+str(i))
        for j in range(0,pt.num_tests):
            print('test: ',j+1,'/',pt.num_tests)
            error_ant[i] = error[i]
            # verifies if the actions are for classification(the last 6 ones)
            if (i>=10):
                params = pt.train_model_c(i)
                print('best_params_' + str(i) + ' = ',params)
                error[i] = pt.evaluate_validation_c(params,i)
                print('accuracy:' + str(i) + ' = ' + str(error[i]))
            else:    
                params = pt.train_model(i)
                print('best_params_' + str(i) + ' = ',params)
                error[i] = pt.evaluate_validation(params,i)
                print('mean_squared_error on validation set:' + str(i) + ' = ' + str(error[i]))
            error_accum[i] += error[i]
            if j == pt.num_tests-1:
                if (i >= 10):
                    print('average accuracy:', error_accum[i]/pt.num_tests)
                else: 
                    print('average error:', error_accum[i]/pt.num_tests)
            # exports model
            if error[i] <= error_ant[i]:    
                pt.export_model(i)
    #for i in range(10,11):
    #    print('Accuracy ', i, ": ", error_accum[i]/pt.num_tests )
       
    