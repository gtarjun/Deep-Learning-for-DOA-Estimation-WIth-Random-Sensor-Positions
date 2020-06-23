# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:01:00 2019

@author: Arjun
"""

#import tensorflow as tf
#import keras
#from tensorflow.keras.models import Sequential 
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Input, Lambda
from keras import optimizers
import pickle
from keras.models import Model


X = pickle.load(open("X_data.pickle","rb")) #load feature data
y = pickle.load(open("Y_data.pickle","rb")) #load target data

x_max = X.max() #find max of feature matrix
X = X/x_max #normalize data

y_max = y.max() #find max target
y = y/y_max #normalize target 

input_shape= Input(shape = X.shape[1:])

#model = Sequential()

#Layer 1 Conv and Pool
conv_1 = Conv2D(64,(8,2), padding='same',activation='relu')(input_shape)
output_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

#Layer2 Conv and Pool
conv_2 = Conv2D(128,(8,2),padding='same',activation='relu')(output_1)
output_2 = MaxPooling2D(pool_size=(2,2))(conv_2)

conv_3 = Conv2D(128,(1,1),padding='valid',activation='relu')(output_2)
#output_3 =  MaxPooling2D(pool_size=(2,2))(conv_3)
conv_4 = Conv2D(8,(1,1),padding='valid',activation='relu')(conv_3)
#Dense Layer
flat_data = Flatten()(conv_4)
dense_1 = Dense(64,activation='relu')(flat_data)
#model.add(Activation("relu"))

#output layer
y_estimated = Dense(1,activation='linear')(dense_1)


model = Model(inputs=input_shape, outputs=[y_estimated])
model.summary()


adam_opt = optimizers.Adam(lr=0.005,decay=1e-6)

model.compile(loss="mean_squared_error", optimizer = adam_opt, metrics=['mse'])

model.fit(X, y,batch_size=32, epochs=10, validation_split = 0.2)





