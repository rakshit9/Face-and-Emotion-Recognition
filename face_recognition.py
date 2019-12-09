#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:59:02 2019

@author: rakshit
"""

import sys
import os 
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
#from keras.utils import np_untils



df =  pd.read_csv('fer2013.csv')

#print(df.info())

#print(df["Usage"].value_counts)

#print(df.head())




x_train,train_y,x_test,test_y = [],[],[],[],[]
