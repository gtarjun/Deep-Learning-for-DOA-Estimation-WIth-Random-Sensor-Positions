#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:59:12 2019

@author: arjun
"""

import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
import pickle
from tensorflow.keras.callbacks import TensorBoard


#load data
X = pickle.load(open("X_data_randsig_aps.pickle","rb")) #load feature data
y = pickle.load(open("Y_data_randsig_aps.pickle","rb")) #load target data

#model architecture
dense_layer = [2]
size_layer = 128
EPOCHS = 1500


model_architecture = "{}-dense-{}-layersize-run3-{}".format(dense_layer,size_layer,int(time.time()))
print(model_architecture)
#input_s= Input(shape = X.shape[1:])
model = Sequential()

model.add(Dense(size_layer, input_shape=X.shape[1:]))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
 
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             


model.add(Dense(20))
model.add(Activation('linear'))
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_architecture))
#checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' #
#checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [tensorboard]

model.summary()
optim = optimizers.Adam(lr=0.005,decay=1e-6)

model.compile(loss='mae',optimizer=optim, metrics=['mse','mae'])
history = model.fit(X, y, batch_size=60, epochs=EPOCHS, validation_split=0.3, callbacks=[tensorboard])
#model.save('{}.hdf5'.format(model_architecture))

        
        
        
