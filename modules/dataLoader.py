#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 23:16:39 2018

@author: Ethan Cheng
"""

from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class behaviorCloneData(object):
    
    def __init__(self, path ):
        self.path = path + '/'
        self.df = pd.read_csv(self.path + 'driving_log.csv')
        self.df_train, self.df_valid = train_test_split(self.df, test_size=0.2)
        self.x_valid, self.y_valid = self.df2xy(self.df_valid)
        
    def df2xy(self, dff, random_flip = False):
        images = []
        y = np.array(dff.steering)
        for i in range(len(dff)):
            im = cv2.imread(self.path + dff.center.iloc[i])
            if (np.random.rand() > 0.5):
                im = im[:,::-1,:]
                y[i] = 1.0 - y[i]
            images.append(im)
        return np.array(images), y
    
    def trainBatchGenerator(self, batch_size=32):
        while 1: # Loop forever so the generator never terminates
            shuffle(self.df_train)
            for offset in range(0, len(self.df_train), batch_size):
                df_i = self.df_train[offset:offset+batch_size]
                yield shuffle(self.df2xy(df_i, True))

    def validationSetGenerator(self):
        while 1:
            yield self.x_valid, self.y_valid
        
        
        
