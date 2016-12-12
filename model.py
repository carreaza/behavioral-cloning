#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:27:19 2016

@author: carlos
"""
import tensorflow as tf
import pickle
import json
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Input, Flatten, Convolution2D, MaxPooling2D
from keras.models import Model
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


###LOAD DATA
image_names = np.genfromtxt('driving_log.csv', dtype='str', delimiter=',', usecols=[0])
labels = np.loadtxt('driving_log.csv',delimiter=',',usecols=[3])

count = 0
features = np.zeros((len(image_names),160,320))
for p in image_names:
    features[count]= cv2.imread(p,0)
    #features[count] = cv2.resize(im,(32,32))
    count +=1

dataset = {'features':features,'labels':labels}
X_train, y_train = dataset['features'],dataset['labels']

###PRE PROCESS DATA

X_train /= 255



###MODEL
nb_filters = 32
kernel_size = (3,3)
image_shape = (160,320)
input_shape = (image_shape[0],image_shape[1],1)
pool_size = (2,2)

X_train = X_train.reshape(X_train.shape[0], image_shape[0], image_shape[1], 1)
#X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
#model.add(Activation('softmax'))

###COMPILE MODEL WITH MSE
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
###TRAIN MODEL
nb_epoch = 8
batch_size = 100


model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
#model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#          verbose=1, validation_data=(X_test, y_test))
#   
    
# serialize model to JSON
model_json = model.to_json()
with open('model.json','w') as f:
    json.dump(model_json,f)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


