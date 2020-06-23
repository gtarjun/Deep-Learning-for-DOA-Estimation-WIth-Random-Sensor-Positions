# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:11:18 2019

@author: Arjun
"""

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten 
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

X = pickle.load(open("X_data_cnn.pickle","rb")) #load feature data
y = pickle.load(open("Y_data_cnn.pickle","rb")) #load target data

dense_layers = [0,1,3,4]
layer_sizes_dense = [64,128,256]
layer_sizes_convolution = [64,128,256]
convolution_layers = [1,2,3,4]
EPOCHS = 10

for dense_layer in dense_layers:
    for size_layer_dense in layer_sizes_dense:
        for convolution_layer in convolution_layers:
            for size_layer_convolution in layer_sizes_convolution:
                model_architecture = "{}denselayer-{}layer_size_dense-{}convlayer-{}layer_size_conv-{}".format(dense_layer,size_layer_dense,convolution_layer,size_layer_convolution,int(time.time()))
                print(model_architecture)
                tensorboard = TensorBoard(log_dir='logs/{}'.format(model_architecture))
                
                model=Sequential()
                model.add(Conv2D(size_layer_convolution,(4,2),input_shape= X.shape[1:], padding='valid'))
                model.add(Activation("relu"))
                #model.add(MaxPooling2D(pool_size=(2,2)))
                
                for l in range (convolution_layer-1):
                    model.add(Conv2D(size_layer_convolution,(4,2), padding='valid'))
                    model.add(Activation("relu"))
                   
                
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
                
                for _ in range(dense_layer):
                    model.add(Dense(size_layer_dense))
                    model.add(Activation("relu"))
                    model.add(Dropout(0.2))
                    
                model.add(Dense(1))
                #model.add(Activation("linear"))
                model.summary()
                
                adam_opt = optimizers.Adam(lr=0.005,decay=1e-6)
                model.compile(loss="mean_absolute_error", optimizer = adam_opt, metrics=['mae','mse'])
                model.fit(X, y,batch_size=32, epochs=EPOCHS, validation_split = 0.3, callbacks=[tensorboard])
                
            
                
                