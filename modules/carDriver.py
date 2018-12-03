#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:39:39 2018

@author: Ethan Cheng
"""

#import pickle
#import numpy as np
#import tensorflow as tf
#from matplotlib import image as mpimg
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Lambda, Conv2D, Cropping2D, MaxPooling2D

params = {
        'input_shape':  (160,320, 3),
        'cropping': ((50,20), (0,0)),
        'nFilters': [24, 36, 48, 64],     
        'kernel': [5,5,5, 3,3], 
        'maxPoolStride' : [2,2,2,2,2],
        'convDropout' : 0.2,
        'FC': [256, 64],
        'fcDropout': 0.5,
        'outDim': 10,
        'batch_size' : 128
        }

#%%
class carDriver(Sequential):
       
    def __init__(self, params):
        Sequential.__init__(self)
        self.__dict__.update(params)
        # Layer #1: normalization
        self.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape= params['input_shape'], output_shape = params['input_shape']))
        self.add(Cropping2D(cropping= params['cropping'], input_shape= params['input_shape']))
        # Layer #2~5: convolution layers
        for i, n in enumerate(self.nFilters):
            self.add(Conv2D(n, (self.kernel[i], self.kernel[i]), padding='same', activation='relu'))
            self.add(BatchNormalization())
            self.add(MaxPooling2D((self.maxPoolStride[i], self.maxPoolStride[i])))
            self.add(Dropout(self.convDropout))
        # flatten layer
        self.add(Flatten())
        # fully connected layers
        for i, n in enumerate(self.FC):
            self.add(Dense(n, activation = 'relu'))
            self.add(BatchNormalization())
            self.add(Dropout(self.fcDropout))
        self.add(Dense(self.outDim, activation = 'softmax'))
        
        #%%
model = carDriver(params)        

#%%

model.compile(loss='mse', optimizer='adam')
