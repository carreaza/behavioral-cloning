#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:27:19 2016

@author: carlos
"""
import json
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Flatten, Convolution2D, MaxPooling2D, Lambda, ELU
from keras.models import Model
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

###Global variables
image_shape = (64,64)
IMG_CH = 3

##FUNCTIONS

###This function preprocesses the images by resizing them, and reshaping 
###them to the appropriate shape needed by Keras 1xWxDxC 
###(W=width, D=depth, C=channels)
def preprocess_image(image):
    
    #remove top 60pixels (sky) and bottom 40 pixels (car's hood)
    image = image[60:140,:,:]
    image = cv2.resize(image,(image_shape[1],image_shape[0]),interpolation=cv2.INTER_AREA)
    image = image.reshape(1, image_shape[0], image_shape[1], IMG_CH)
    return image

###This function changes the brightness of the image ramdonly to augment the data    
def change_brightness(image):
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute a random brightness value and apply to the image
    brightness = 0.25 + np.random.uniform()
    temp[:, :, 2] = temp[:, :, 2] * brightness
    # Convert back to RGB and return
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB) 

###This function flips the image horizontally to augment the data    
def flip_image(image, steering):
    if np.random.randint(2) == 0:
        image = np.fliplr(image)
        steering = -steering 
    return image, steering
    
###This function is used in fit_generatior() and yields a single image at a time    
def batch_generator(image_names, Y):
    while 1:
        for i in range(len(image_names)):
            steering = np.array([[Y[i]]])
            image = cv2.imread(image_names[i])
            
            ###Flipping the image
            image, steering = flip_image(image,steering)
                
            #change brightness
            image = change_brightness(image)
                
            #resize, reshape image for keras model    
            image = preprocess_image(image)
            yield image,steering
  
###LOAD DATA
image_names = np.genfromtxt('driving_log.csv', dtype='str', delimiter=',', usecols=[0])
labels = np.loadtxt('driving_log.csv',delimiter=',',usecols=[3])

###PRE PROCESS DATA
###Separate for training and validation
X_train, X_val, y_train, y_val = train_test_split(image_names,labels, random_state=41, test_size=0.33)

###MODEL

input_shape = (image_shape[0],image_shape[1],IMG_CH)
pool_size = (2,2)

model = Sequential()
model.add(Lambda(lambda x: x/255. - 0.5,input_shape=input_shape, output_shape=input_shape))
model.add(Convolution2D(32, 3, 3,border_mode='same', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(32, 3, 3,border_mode='same', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3,border_mode='same', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(64, 3, 3,border_mode='same', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Convolution2D(128, 3, 3,border_mode='same', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(128, 3, 3,border_mode='same', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, init='he_normal'))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(64, init='he_normal'))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(16, init='he_normal'))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(1, init='he_normal'))
model.summary()

###COMPILE MODEL WITH MSE
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

###TRAIN MODEL
history = model.fit_generator(batch_generator(X_train, y_train), samples_per_epoch = X_train.shape[0], nb_epoch = 8,
                    verbose=1, callbacks=[], validation_data=batch_generator(X_val, y_val),
                    nb_val_samples=500)
    
# serialize model to JSON, write weights and model arch to output files
model_json = model.to_json()
with open('model.json','w') as f:
    json.dump(model_json,f)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


