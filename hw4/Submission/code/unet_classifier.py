import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import argparse
import pickle
import keras

from tensorflow.keras.models import Model

from keras import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Convolution2D, Dense, MaxPooling2D, GlobalMaxPooling2D, Flatten, BatchNormalization, Dropout, SpatialDropout2D, Activation, UpSampling2D, Concatenate

import random
import re

import png

import sklearn.metrics


# inception module to check best model fit, though avoid due to long computation times.
def inception_module(input_tensor, nfilters, activation, p_spatial_dropout, regularization, i):
##### A #####
    convA_tensor = Convolution2D(filters=nfilters[0],
                                kernel_size=(1,1),
                                strides=(2,2),
                                padding='same',
                                kernel_regularizer=regularizer,
                                name='convA_%d'%i)(input_tensor)

    convA_tensor = BatchNormalization()(convA_tensor)
    convA_tensor = Activation(activation)(convA_tensor)
    convA_tensor = SpatialDropout2D(p_spatial_dropout)(convA_tensor)

##### B #####

    convB0_tensor = Convolution2D(filters=nfilters[1][0],
                                kernel_size=(1,1),
                                strides=(1,1),
                                padding='same',
                                kernel_regularizer=regularizer,
                                name='convB0_%d'%i)(input_tensor)

    convB0_tensor = BatchNormalization()(convB0_tensor)
    convB0_tensor = Activation(activation)(convB0_tensor)
    convB0_tensor = SpatialDropout2D(p_spatial_dropout)(convB0_tensor)

    convB1_tensor = Convolution2D(filters=nfilters[1][1],
                                kernel_size=(3,3),
                                strides=(2,2),
                                padding='same',
                                kernel_regularizer=regularizer,
                                name='convB1_%d'%i)(convB0_tensor)

    convB1_tensor = BatchNormalization()(convB1_tensor)
    convB1_tensor = Activation(activation)(convB1_tensor)
    convB1_tensor = SpatialDropout2D(p_spatial_dropout)(convB1_tensor)

    convC0_tensor = Convolution2D(filters=nfilters[2][0],
                                kernel_size=(1,1),
                                strides=(1,1),
                                padding='same',
                                kernel_regularizer=regularizer,
                                name='convC0_%d'%i)(input_tensor)

    convC0_tensor = BatchNormalization()(convC0_tensor)
    convC0_tensor = Activation(activation)(convC0_tensor)
    convC0_tensor = SpatialDropout2D(p_spatial_dropout)(convC0_tensor)

##### C #####

    convC1_tensor = Convolution2D(filters=nfilters[2][1],
                                kernel_size=(5,5),
                                strides=(2,2),
                                padding='same',
                                kernel_regularizer=regularizer,
                                name='convC1_%d'%i)(convC0_tensor)

    convC1_tensor = BatchNormalization()(convC1_tensor)
    convC1_tensor = Activation(activation)(convC1_tensor)
    convC1_tensor = SpatialDropout2D(p_spatial_dropout)(convC1_tensor)

    max_tensor = MaxPooling2D(pool_size=(3,3),
                               strides=(1,1),
                               padding='same',
                               name='MP%d'%i)(tensor)
 
##### D #####

    convD1_tensor = Convolution2D(filters=nfilters[3],
                                kernel_size=(1,1),
                                strides=(2,2),
                                padding='same',
                                kernel_regularizer=regularizer,
                                name='convD1_%d'%i)(max_tensor)

    convD1_tensor = BatchNormalization()(convD1_tensor)
    convD1_tensor = Activation(activation)(convD1_tensor)
    convD1_tensor = SpatialDropout2D(p_spatial_dropout)(convD1_tensor)

    output_tensor = Add()([convA_tensor, convB1_tensor, convC1_tensor, convD1_tensor])
    #output_tensor = Concatenate()([convA_tensor, convB1_tensor, convC1_tensor, convD1_tensor])
    return output_tensor

#################################################################################################


# Encoder Module
def build_layer_down(input_tensor, n, conv_activation, p_spatial_dropout, padding, regularizer, i):
    tensor = Convolution2D(filters=n['filters'],
                            kernel_size=n['kernel_size'],
                            strides=(1,1),
                            padding=padding,
                            kernel_regularizer=regularizer,
                            name='Down_C1_%d'%i)(input_tensor)

    if n['batch_normalization']:
        tensor = BatchNormalization()(tensor)
        
    tensor = Activation(conv_activation)(tensor)

    if p_spatial_dropout is not None:
        tensor = SpatialDropout2D(p_spatial_dropout)(tensor)

#----------

    tensor = Convolution2D(filters=2*n['filters'],
                            kernel_size=n['kernel_size'],
                            strides=(1,1),
                            padding=padding,
                            kernel_regularizer=regularizer,
                            name='Down_C2_%d'%i)(tensor)

    if n['batch_normalization']:
        tensor = BatchNormalization()(tensor)
        
    tensor = Activation(conv_activation)(tensor)

    if p_spatial_dropout is not None:
        tensor = SpatialDropout2D(p_spatial_dropout)(tensor)

#----------

    # Pooling
    if n['pool_size'] is not None:
        tensor = MaxPooling2D(pool_size=n['pool_size'],
                               strides=n['strides'],
                               padding=padding,
                               name='MP_%d'%i)(tensor)

    return tensor

#=========


# Decoder Module
def build_layer_up(input_tensor, n, conv_activation, p_spatial_dropout, padding, regularizer, i):
    tensor = Convolution2D(filters=n['filters'],
                            kernel_size=n['kernel_size'],
                            strides=(1,1),
                            padding=padding,
                            kernel_regularizer=regularizer,
                            name='Up_C1_%d'%i)(input_tensor)

    if n['batch_normalization']:
        tensor = BatchNormalization()(tensor)
        
    tensor = Activation(conv_activation)(tensor)

    if p_spatial_dropout is not None:
        tensor = SpatialDropout2D(p_spatial_dropout)(tensor)

#----------

    tensor = Convolution2D(filters=2*n['filters'],
                            kernel_size=n['kernel_size'],
                            strides=(1,1),
                            padding=padding,
                            kernel_regularizer=regularizer,
                            name='Up_C2_%d'%i)(tensor)

    if n['batch_normalization']:
        tensor = BatchNormalization()(tensor)
        
    tensor = Activation(conv_activation)(tensor)

    if p_spatial_dropout is not None:
        tensor = SpatialDropout2D(p_spatial_dropout)(tensor)
#----------

    # Upsample
    tensor = UpSampling2D(size=n['strides'], name="UpSample_%d"%i)(tensor)

    return tensor

#=========


def create_unet(image_size,
                                  nchannels, 
                                  conv_layers,
                                  p_spatial_dropout,
                                  lambda_l2,  # None or a float
                                  lrate, 
                                  n_classes,
                                  loss,
                                  metrics,
                                  padding,
                                  flatten,
                                  conv_activation):


    if lambda_l2 is not None:
        # assume a float
        regularizer = tf.keras.regularizers.l2(lambda_l2)
    else:
        regularizer = None
    
    tensor_stack = []
    input_tensor = Input(shape=(image_size[0], image_size[1], nchannels))
    tensor = input_tensor

    # Using inception_module for skip connections (not for unet though)
    #tensor = inception_module(tensor, (10, (10,10), (10,10), 10), activation, p_spatial_dropout, regularization, 0)
#def inception_module(input_tensor, nfilters, activation, SpatialDropout2D, regularization, name):


    # Down
    for i in range(len(conv_layers)):
        print('1', tensor.shape)
        tensor = build_layer_down(tensor, conv_layers[i], conv_activation, p_spatial_dropout, padding, regularizer, i)
        tensor_stack.append(tensor)

    # Up
    for i in reversed(conv_layers):
        print('2', tensor_stack[-1].shape)
        print(conv_layers[i]['strides'])
        print('3', tensor.shape)

        skip_connection = tensor_stack.pop()

        tensor = Concatenate()([tensor, skip_connection])
        tensor = build_layer_up(tensor, conv_layers[i], conv_activation, p_spatial_dropout, padding, regularizer, i)

    tensor = Convolution2D(n_classes, (1, 1), activation="softmax", name='output')(tensor)  # For single-channel output

    model = Model(inputs=input_tensor, outputs=tensor)
    
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate = lrate,
                                   clipnorm=1.0,
                                   amsgrad = False)
    
    # Bind the model to the optimizer
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics)

    return model
        

