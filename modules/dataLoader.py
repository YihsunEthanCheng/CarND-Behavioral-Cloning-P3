#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 23:16:39 2018

@author: Ethan Cheng
"""

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class behaviorCloneData(object):
    
    def __init__(self, path ):
        self.path = path + '/'
        self.df = pd.read_csv(self.path + 'driving_log.csv')
        self.df_train, self.df_valid = train_test_split(self.df, test_size=0.2)
        self.x_valid, self.y_valid = self.df2xy(self.df_valid)
        self.trainSize = len(self.df_train)
        self.validSize = len(self.df_valid)
        
    def df2xy(self, dff, random_flip = False):
        images = []
        y = [] #np.array(dff.steering)
        for i in range(len(dff)):
            try:
                im = mpimg.imread(self.path + dff.center.iloc[i])
            except:
                raise ValueError('Error reading image {}'.format(self.path + dff.center.iloc[i]))
            
            yi = dff.steering.iloc[i]
            if random_flip and np.random.rand() > 0.5:
                im = np.fliplr(im) 
                yi = -yi
            images.append(im)
            y += [yi]
        return np.array(images), np.array(y)
    
    def trainBatchGenerator(self, batch_size=32, random_flip = True):
        while 1: # Loop forever so the generator never terminates
            shuffle(self.df_train)
            for offset in range(0, batch_size*(len(self.df_train)//batch_size), batch_size):
                yield self.df2xy(self.df_train[offset:offset+batch_size], random_flip)
        
        
        
        
        
