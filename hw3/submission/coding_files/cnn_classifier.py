import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import argparse
import pickle
import keras

from keras import Sequential
from keras.layers import InputLayer
from keras.layers import Convolution2D, Dense, MaxPooling2D, GlobalMaxPooling2D, Flatten, BatchNormalization, Dropout, SpatialDropout2D, Activation

import random
import re

import png

from core50 import *

import sklearn.metrics


def create_cnn_classifier_network(image_size,
                                  nchannels, 
                                  conv_layers,
                                  dense_layers,
                                  p_dropout,
                                  p_spatial_dropout,
                                  lambda_l2,  # None or a float
                                  lrate, 
                                  n_classes,
                                  loss,
                                  metrics,
                                  padding,
                                  flatten,
                                  conv_activation,
                                  dense_activation):

             #                 n_filters=[10],
             #                 kernel_size=[3],
             #                 pooling=[1],
             #                n_hidden=[5]):
    
    if lambda_l2 is not None:
        # assume a float
        regularizer = tf.keras.regularizers.l2(lambda_l2)
    else:
        regularizer = None
        
    
    # TODO
    model = Sequential()
    model.add(InputLayer(shape=(image_size[0], image_size[1], nchannels)))
   
    # convolutional layers
    #for i, (n, s, p) in enumerate(zip(n_filters, kernel_size, pooling)):
    #for i, (n, s, p, st) in enumerate(zip(conv_layers['filters'], conv_layers['kernel_size'], conv_layers['pool_size'], conv_layers[0]['strides'])):
    for i,n in enumerate(conv_layers):
        model.add(Convolution2D(filters=n['filters'],
                                kernel_size=n['kernel_size'],
                                strides=n['strides'],
                                padding=padding,
                                kernel_regularizer=regularizer,
                                name='C%d'%i))

        model.add(Activation(conv_activation))

        if n['batch_normalization']:
            model.add(BatchNormalization())
        

        if p_spatial_dropout is not None:
            model.add(SpatialDropout2D(p_spatial_dropout))

        # Pooling
        #if n['pool_size'][0] > 1:
        if n['pool_size'] is not None:
            model.add(MaxPooling2D(pool_size=n['pool_size'],
                                   strides=n['strides'],
                                   padding=padding,
                                   name='MP%d'%i))

    
    # Flatten
    if flatten:
        model.add(Flatten())

    model.add(GlobalMaxPooling2D())
    
    # Dense layers
    for i,n in enumerate(dense_layers):
        model.add(Dense(units=n['units'],
                        kernel_regularizer=regularizer,
                        name='D%d'%i))

        if n['batch_normalization']:
            model.add(BatchNormalization())

        model.add(Activation(dense_activation))

        if p_dropout is not None:
            model.add(Dropout(p_dropout))
        


    # Output
    model.add(Dense(units=n_classes,
                    activation='softmax',
                    name='output'))
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = lrate,
                                   amsgrad = False)
    
    # Bind the model to the optimizer
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics)
    
    return model


