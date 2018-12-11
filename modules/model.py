#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:23:02 2018

@author: Ethan Cheng
"""
#%load_ext autoreload
#%autoreload 2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Lambda, Conv2D, Cropping2D, MaxPooling2D
import matplotlib.pyplot as plt
from modules.dataLoader import behaviorCloneData
from time import time
import numpy as np

params = {
        'input_shape':  (160,320, 3),
        'cropping': ((65,20), (0,0)),
        'nFilters': [24,36,48,60],
        'convDropout' : 0.1,
        'nFC': [128, 16],
        'fcDropout': 0.4,
        'batch_size' : 64
        }

def createModel(params = params):
    model = Sequential()
    # Layer #1: normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape= params['input_shape']))
    model.add(Cropping2D(cropping= params['cropping'], input_shape= params['input_shape']))
    
    # conv #1
    model.add(Conv2D(params['nFilters'][0], (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(params['convDropout']))
    # conv #1
    model.add(Conv2D(params['nFilters'][1], (5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(params['convDropout']))
    # conv #1
    model.add(Conv2D(params['nFilters'][2], (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(params['convDropout']))
    # conv #1
    model.add(Conv2D(params['nFilters'][3], (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(params['convDropout']))
    # flatten layer
    model.add(Flatten())
    # FC #1
    model.add(Dense(params['nFC'][0], activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(params['fcDropout']))
    # FC #2
    model.add(Dense(params['nFC'][1], activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(params['fcDropout']))
    # FC #3
    model.add(Dense(1, activation = 'tanh'))
    
    # add optimizer 
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model
    
def train(model, data, nEpoch = 10, fname  = ''):
    start = time()
    model.fit_generator(
        data.trainBatchGenerator(params['batch_size']), 
        steps_per_epoch = data.trainSize//params['batch_size'],
        epochs = nEpoch,
        validation_data = (data.x_valid, data.y_valid),
        verbose=2)
    yhat = model.predict(data.x_valid,  verbose=1)
    y_valid = data.y_valid.reshape(len(data.y_valid),1)
    plt.clf()
    plt.plot(yhat,'r.')
    plt.plot(y_valid,'b.')
    if fname != '':
        model.save(fname)
    print("Training time: {} sec".format(time()-start))
