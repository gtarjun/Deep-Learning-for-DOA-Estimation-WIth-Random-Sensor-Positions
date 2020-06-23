#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:12:36 2019
Alexnet for CNN regression DOA estimation
@author: arjun
"""
import time
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Input, Lambda
from keras import optimizers
import pickle
from keras.models import Model
from keras.callbacks import TensorBoard

X = pickle.load(open("X_data_20sensor_1layer_deg.pickle","rb")) #load feature data
y = pickle.load(open("Y_data_20sensor_1layer_deg.pickle","rb")) #load target data

EPOCHS = 300
dense_layer = [1]
size_layer_dense = [128]
size_layer_convolution = [256]
convolution_layer = [5]
Iterations = 100000
N = 40
L = 20

input_shape= Input(shape = X.shape[1:])

#Layer 1 Conv and Pool
conv_1 = Conv2D(128,(2,2),padding='same',activation='relu')(input_shape)
output_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

#Layer2 Conv and Pool
conv_2 = Conv2D(256,(3,3),padding='same',activation='relu')(output_1)
#output_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
#Standalone Convolution layers
conv_3 = Conv2D(384,(2,2),padding='valid',activation='relu')(conv_2)
conv_4 = Conv2D(384,(2,2),padding='valid',activation='relu')(conv_3)
conv_5 = Conv2D(256,(2,2),padding='valid',activation='relu')(conv_4)
output_3 = MaxPooling2D(pool_size=(2,2))(conv_5)

flat_data = Flatten()(output_3)   

#Dense Layer
dense_1 = Dense(256,activation='relu')(flat_data)
dropout_1 = Dropout(0.3)(dense_1)
dense_2 = Dense(128,activation='relu')(dropout_1)
dropout_2 = Dropout(0.3)(dense_2)

#output layer
y_estimated = Dense(1,activation='linear')(dropout_2)


model = Model(inputs=input_shape, outputs=[y_estimated])
model.summary()

model_architecture = "{}denselayer-{}layer_size_dense-{}convlayer-{}layer_size_conv-{}lexnet-deg".format(dense_layer,size_layer_dense,convolution_layer,size_layer_convolution,int(time.time()))
print(model_architecture)
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_architecture))

adam_opt = optimizers.Adam(lr=0.005,decay=1e-6)

model.compile(loss="mean_squared_error", optimizer = adam_opt, metrics=['mse', 'mae'])

model.fit(X, y,batch_size=32, epochs=EPOCHS, validation_split = 0.3)




