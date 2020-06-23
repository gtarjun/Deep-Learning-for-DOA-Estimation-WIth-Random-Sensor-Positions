#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:59:12 2019

@author: arjun
"""

import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import Model
from tensorflow.keras import optimizers
import pickle
from tensorflow.keras.callbacks import TensorBoard


#load data
X = pickle.load(open("X_data_randsig_aps.pickle","rb")) #load feature data
y = pickle.load(open("Y_data_randsig_aps.pickle","rb")) #load target data

#model architecture
dense_layer = [2]                                                                                                                                                                                                                                                                                                                                                                                                                                               
size_layer1 = 128
size_layer2 = 256
EPOCHS = 1000
class SensorInterpolator(Model):
        
    def __init__(self):
        super(SensorInterpolator, self).__init__()
        self.interpolator_input = Dense(size_layer1, input_shape=X.shape[1:], activation='relu')
        self.dense1 = Dense(size_layer1, activation='relu')
        self.dense2 = Dense(size_layer2, activation='relu')
        self.droput = Dropout(0.2)
        self.interpolator_output = Dense(20, activation='linear')
    
    def call(self,X):
        interpolator_input = self.interpolator_input(X)
        layer1 = self.dense1(interpolator_input)
        dropout1 =  self.droput(layer1)
        layer2 = self.dense2(dropout1)
        dropout2 = self.droput(layer2)
        
        return self.interpolator_output(dropout2)
        
        
        
model = SensorInterpolator()     
optim = optimizers.Adam(lr=0.000001,decay=1e-9)        

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optim.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


        
model_architecture = "{}-dense-{}-layersize-run3-{}".format(dense_layer, size_layer1, size_layer2,int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_architecture))
#checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' #
#checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [tensorboard]

model.summary()


model.compile(loss='mae',optimizer=optim, metrics=['mse','mae'])
history = model.fit(X, y, batch_size=60, epochs=EPOCHS, validation_split=0.3, callbacks=[tensorboard])
model.save('{}.hdf5'.format(model_architecture))

        
        
        
#
#model = Sequential()
#
#model.add(Dense(size_layer, input_shape=X.shape[1:]))
#model.add(Activation('relu'))
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
# 
#model.add(Dense(256))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
#
##model.add(Dense(256))
##model.add(Activation('relu'))
##model.add(Dropout(0.2))
#
#model.add(Dense(20))
#model.add(Activation('linear'))