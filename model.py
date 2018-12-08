#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:23:02 2018

@author: Ethan Cheng
"""
%load_ext autoreload
%autoreload 2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Lambda, Conv2D, Cropping2D, MaxPooling2D

params = {
        'input_shape':  (160,320, 3),
        'cropping': ((50,20), (0,0)),
        'convDropout' : 0.2,
        'fcDropout': 0.5,
        'batch_size' : 64
        }


#%%
from modules.dataLoader import behaviorCloneData

data = behaviorCloneData('data/teach1')

#%%
model = Sequential()
# Layer #1: normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape= params['input_shape']))
model.add(Cropping2D(cropping= params['cropping'], input_shape= params['input_shape']))

# conv #1
model.add(Conv2D(24, (5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(params['convDropout']))
# conv #1
model.add(Conv2D(36, (5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(params['convDropout']))
# conv #1
model.add(Conv2D(48, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(params['convDropout']))
# conv #1
model.add(Conv2D(64, (2,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(params['convDropout']))
# flatten layer
model.add(Flatten())
# FC #1
model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(params['fcDropout']))
# FC #2
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(params['fcDropout']))
model.add(Dense(1, activation = 'sigmoid'))

# add optimizer 
model.compile(loss='mse', optimizer='adam')
model.summary()

#%%
nEpoch = 25
model.fit_generator(
    data.trainBatchGenerator(params['batch_size']), 
    steps_per_epoch = data.trainSize//params['batch_size'],
    epochs = nEpoch,
    validation_data = (data.x_valid, data.y_valid),
    verbose=2)

