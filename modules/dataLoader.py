#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 23:16:39 2018

@author: Ethan Cheng
"""

from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
import pandas as pd
from sklearn.utils import shuffle

#%%

df = pd.read_csv('./data/driving_log.csv')
df_train, df_valid = train_test_split(df, test_size=0.2)


def generator(df, batch_size=32, dataPath = 'data/'):
    while 1: # Loop forever so the generator never terminates
        shuffle(df)
        for offset in range(0, len(df), batch_size):
            batch = df[offset:offset+batch_size]
            images = []
            for i in range(len(batch)):
                images.append(cv2.imread(dataPath + batch.center[i]))
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(batch.steering)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(df_train, batch_size=32)