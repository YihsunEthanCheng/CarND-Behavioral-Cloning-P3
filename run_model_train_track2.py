#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:58:24 2018

@author: Ethan Cheng
"""
%load_ext autoreload
%autoreload 2
from modules.model import createModel, train
from modules.dataLoader import behaviorCloneData


params = {
        'input_shape':  (160,320, 3),
        'cropping': ((65,20), (0,0)),
        'nFilters': [32,48,64, 80],
        'convDropout' : 0.1,
        'nFC': [256, 32],
        'fcDropout': 0.4,
        'batch_size' : 128
        }

#%%
trackID = 2
data = behaviorCloneData('data/track{}'.format(trackID))
model = createModel(params)

#%%
accumNepoch = 0
nEpoch = 10
train(model, data, nEpoch, 
    'trained_model/track{}/{}_{}_ep{}.h5'.format(trackID,
    '-'.join(np.array(params['nFilters']).astype(str)),
    '-'.join(np.array(params['nFC']).astype(str)), nEpoch +accumNepoch))
accumNepoch +=  nEpoch
#%% repeat this cell until signs of overtraining show up
nEpoch = 10
train(model, data, nEpoch, 
    'trained_model/track{}/{}_{}_ep{}.h5'.format(trackID,
    '-'.join(np.array(params['nFilters']).astype(str)),
    '-'.join(np.array(params['nFC']).astype(str)), nEpoch+ accumNepoch))
accumNepoch += nEpoch

#