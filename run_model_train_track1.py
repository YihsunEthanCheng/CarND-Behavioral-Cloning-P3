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
        'nFilters': [24,36,48,60],
        'convDropout' : 0.1,
        'nFC': [128, 16],
        'fcDropout': 0.4,
        'batch_size' : 64
        }


trackID = 1
data = behaviorCloneData('data/track{}'.format(trackID))
model = createModel(params)
#%%
nEpoch = 25
train(model, data, nEpoch, 
    'trained_model/track{}/{}_{}_ep{}.h5'.format(trackID,
    '-'.join(np.array(params['nFilters']).astype(str)),
    '-'.join(np.array(params['nFC']).astype(str)), nEpoch))