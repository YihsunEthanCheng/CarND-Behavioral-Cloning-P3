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

#%%

def df2xy(dff, dataPath = 'data/'):
    images = []
    for i in range(len(dff)):
        images.append(cv2.imread(dataPath + dff.center.iloc[i]))
    return np.array(images), np.array(dff.steering)

def generator(df, batch_size=32, dataPath = 'data/'):
    while 1: # Loop forever so the generator never terminates
        shuffle(df)
        for offset in range(0, len(df), batch_size):
            df_i = df[offset:offset+batch_size]
            X_train, y_train = df2xy(df_i, dataPath)
            yield shuffle(X_train, y_train)

#%%
df = pd.read_csv('./data/driving_log.csv')
df_train, df_valid = train_test_split(df, test_size=0.2)

X_valid, y_valid =  df2xy(df_valid[:10])

# compile and train the model using the generator function
train_generator = generator(df_train, batch_size=32)

X_train, y_train = next(train_generator)
