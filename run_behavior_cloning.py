#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:01:42 2018

@author: Ethan Cheng
"""
%load_ext autoreload
%autoreload 2
from modules.dataLoader import behaviorCloneData
from modules.roboPilot import pilot

#%%
params = {
        'input_shape':  (160,320, 3),
        'cropping': ((50,20), (0,0)),
        'nFilters': [24, 36, 48, 64],     
        'kernel': [5,5,5,3,3], 
        'maxPoolStride' : [2,2,2,2,2],
        'convDropout' : 0.2,
        'FC': [256, 64],
        'fcDropout': 0.5,
        'outDim': 10,
        'batch_size' : 128
        }
        
#%%

data = behaviorCloneData('data/download')
x_valid, y_valid = data.validSet()
train_generator = data.batchGenerator(params['batch_size'])

#%%

model = pilot(params)


model.compile(loss='mse', optimizer='adam')
