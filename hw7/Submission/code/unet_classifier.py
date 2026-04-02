import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import argparse
import pickle
import keras


from diffusion_tools import PositionEncoder, convert_image
from keras.models import Model

from keras import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Convolution2D, Dense, MaxPooling2D, AveragePooling2D, Flatten, BatchNormalization, Dropout, SpatialDropout2D, Activation, UpSampling2D, Concatenate

import random
import re

import png

import sklearn.metrics


class ExpandDimsTwice(keras.layers.Layer):
    def call(self, inputs):
        x = tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)
        return x

class TileToImage(keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width

    def call(self, inputs):
        return tf.tile(inputs, [1, self.height, self.width, 1])




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


def create_unet(patch_size,
                  nchannels, 
                  conv_layers,
                  dense_layers,
                  p_spatial_dropout,
                  p_dropout,
                  lambda_l2,  # None or a float
                  lrate, 
                  n_classes,
                  loss,
                  metrics,
                  padding,
                  flatten,
                  conv_activation,
                  dense_activation,
                  embedding_size,
                  time):


    if lambda_l2 is not None:
        # assume a float
        regularizer = keras.regularizers.l2(lambda_l2)
    else:
        regularizer = None
    
    tensor_stack = []
    #input_tensor = Input(shape=(image_size[0], image_size[1], nchannels))

    label_input = keras.Input(shape=(patch_size, patch_size, 7), name="label_input")
    image_input = keras.Input(shape=(patch_size, patch_size, 3), name="image_input")
    time_input  = keras.Input(shape=(1,), dtype=tf.int32, name="time_input")


    time_embedding = PositionEncoder(max_steps=time,
                                     max_dims=embedding_size)(time_input) 

    ## Dense layers
    #for i,n in enumerate(dense_layers):
    #    time_embedding = Dense(units=n['units'],
    #                  kernel_regularizer=regularizer,
    #                  name='D%d'%i)(time_embedding)
    #  
    #    if n['batch_normalization']:
    #        time_embedding = BatchNormalization()(time_embedding)
    #        time_embedding = Activation(dense_activation)(time_embedding)
    #           
    #        if p_dropout is not None:
    #            time_embedding = Dropout(p_dropout)(time_embedding)


    time_embedding = ExpandDimsTwice()(time_embedding)
    time_embedding = TileToImage(patch_size, patch_size)(time_embedding)

    #time_embedding = tf.expand_dims(tf.expand_dims(time_embedding, 1), 1) # 3 dims embedding
    ## Apply Embedding to each of the pixels
    #time_embedding = tf.tile(time_embedding, [1, image_size[0], image_size[1], 1])

    # Combine Inputs
    tensor = Concatenate(axis=-1)([label_input, image_input, time_embedding])

    pooled_labels = []
    pooled_times = []

    pooled_label = label_input
    pooled_time = time_embedding

    # Down
    for i in range(len(conv_layers)):
        print('1', tensor.shape)
        tensor = build_layer_down(tensor, conv_layers[i], conv_activation, p_spatial_dropout, padding, regularizer, i)
        tensor_stack.append(tensor)

        # Pool label and time at each level
        pool_size = conv_layers[i]['pool_size']

        pooled_label = AveragePooling2D(pool_size)(pooled_label)
        pooled_time = AveragePooling2D(pool_size)(pooled_time)

        pooled_labels.append(pooled_label)
        pooled_times.append(pooled_time)

    # Up
    for i in reversed(range(len(conv_layers))):
        print('2', tensor_stack[-1].shape)
        print(conv_layers[i]['strides'])
        print('3', tensor.shape)

        skip_connection = tensor_stack.pop()

        pooled_label = pooled_labels[i]
        pooled_time = pooled_times[i]

        tensor = Concatenate()([tensor, pooled_label, pooled_time, skip_connection])

        #tensor = Concatenate()([tensor, skip_connection])
        tensor = build_layer_up(tensor, conv_layers[i], conv_activation, p_spatial_dropout, padding, regularizer, i)

    ## Dense layers
    #for i,n in enumerate(dense_layers):
    #    tensor = Dense(units=n['units'],
    #                  kernel_regularizer=regularizer,
    #                  name='D%d'%i)(tensor)
    #  
    #    if n['batch_normalization']:
    #        tensor = BatchNormalization()(tensor)
    #        tensor = Activation(dense_activation)(tensor)
    #           
    #        if p_dropout is not None:
    #            tensor = Dropout(p_dropout)(tensor)


    # Scale Down for diffusion NOTE: hard coded
    tensor = Convolution2D(64, (3, 3), activation=conv_activation, padding='same')(tensor) 
    tensor = Convolution2D(32, (3, 3), activation=conv_activation, padding='same')(tensor) 

    tensor = Convolution2D(3, (1, 1), activation=None, name='output')(tensor)  # For single-channel output
    #tensor = Convolution2D(3, (1, 1), activation='linear', name='output')(tensor)  # For single-channel output

    model = Model(inputs=[image_input, label_input, time_input], outputs=tensor)
    
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate = lrate,
                                   clipnorm=1e-2,
                                   amsgrad = False)
    
    # Bind the model to the optimizer
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics)

    return model
        

