import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import argparse
import pickle
import keras

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU

import random
import re

import sklearn.metrics
from positional_encoder import *


def create_simple_RNN(
                                  loss,
                                  metrics,
                                  string_len=3934,
                                  simpleRNN_layers=[50],
                                  nfilters=10,
                                  kernel_size=2, # kernel size and strides should be the same here
                                  strides=2,
                                  dense_layers=[100],
                                  dropout=0.1,
                                  lambda_l2=None,  # None or a float
                                  lrate=1e-4, 
                                  n_classes=46,
                                  conv_activation='relu',
                                  simpleRNN_activation='tanh',
                                  dense_activation='relu',
                                  batchNormalization=True,
                                  layerNormalization=True,
                                  n_tokens=100,
                                  n_embeddings=20, # number of embeddings should be smaller than number of unique tokens (n_tokens)
                                  unroll=False # True for GPU usage, false otherwise
):

    
    if lambda_l2 is not None:
        # assume a float
        regularizer = tf.keras.regularizers.l2(lambda_l2)
    else:
        regularizer = None

    tensor = keras.Input(shape=(string_len,))

    # dim (batch, len)

    input_tensor = tensor

    # Embedding layer
    tensor = keras.layers.Embedding(n_tokens,
                              n_embeddings,
                              input_length = string_len)(tensor)

    # dim (batch, len, n_embeddings)

    # Conv1D layer
    tensor = keras.layers.Conv1D(filters=nfilters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding='same',
                                 kernel_regularizer=regularizer,
                                 activation=conv_activation)(tensor)


    for i,n in enumerate(simpleRNN_layers):
        # SimpleRNN layer
        tensor = keras.layers.SimpleRNN(units=n,
                                        activation=simpleRNN_activation,
                                        dropout=dropout,
                                        kernel_regularizer=regularizer,
                                        unroll=unroll, # True for GPU usage, false otherwise
                                        #recurrent_dropout
                                        return_sequences=True)(tensor)
        if layerNormalization:
            tensor = keras.layers.LayerNormalization()(tensor)
        tensor = keras.layers.Activation(dense_activation)(tensor)

        # 1D average pooling
        tensor = keras.layers.AveragePooling1D(pool_size=kernel_size,
                                   strides=strides,
                                   padding='same')(tensor)

    # SimpleRNN layer (end sequence)
    tensor = keras.layers.SimpleRNN(units=n,
                                    activation=simpleRNN_activation,
                                    dropout=dropout,
                                    kernel_regularizer=regularizer,
                                    unroll=unroll, # True for GPU usage, false otherwise
                                    #recurrent_dropout
                                    return_sequences=False)(tensor)

    for i,n in enumerate(dense_layers):
        # Dense layer
        tensor = keras.layers.Dense(units=n,
                                    activation=dense_activation,
                                    use_bias=True,
                                    kernel_regularizer=regularizer)(tensor)
        if batchNormalization:
            tensor = keras.layers.BatchNormalization()(tensor)
        tensor = keras.layers.Activation(dense_activation)(tensor)

    # Dense layer output softmax
    output_tensor = keras.layers.Dense(units=n_classes,
                                    activation='softmax',
                                    use_bias=True)(tensor)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=lrate,
                                amsgrad=False,
                                clipnorm=1e-2)

    model.compile(optimizer=opt,
                        metrics=metrics,
                        loss=loss)

    return model

########################################################################### 

def create_GRU(
                                  loss,
                                  metrics,
                                  string_len=3934,
                                  GRU_layers=[50],
                                  nfilters=10,
                                  kernel_size=2, # kernel size and strides should be the same here
                                  strides=2,
                                  dense_layers=[100],
                                  dropout=0.1,
                                  lambda_l2=None,  # None or a float
                                  lrate=1e-4, 
                                  n_classes=46,
                                  conv_activation='relu',
                                  GRU_activation='tanh',
                                  dense_activation='relu',
                                  batchNormalization=True,
                                  layerNormalization=True,
                                  n_tokens=100,
                                  n_embeddings=20, # number of embeddings should be smaller than number of unique tokens (n_tokens)
                                  unroll=False # True for GPU usage, false otherwise
):

    
    if lambda_l2 is not None:
        # assume a float
        regularizer = tf.keras.regularizers.l2(lambda_l2)
    else:
        regularizer = None

    tensor = keras.Input(shape=(string_len,))

    # dim (batch, len)

    input_tensor = tensor

    # Embedding layer
    tensor = keras.layers.Embedding(n_tokens,
                              n_embeddings,
                              input_length = string_len)(tensor)

    # dim (batch, len, n_embeddings)

    # Conv1D layer
    tensor = keras.layers.Conv1D(filters=nfilters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding='same',
                                 kernel_regularizer=regularizer,
                                 activation=conv_activation)(tensor)


    for i,n in enumerate(GRU_layers):
        # GRU layer
        tensor = keras.layers.GRU(units=n,
                                        activation=GRU_activation,
                                        recurrent_activation='sigmoid',
                                        dropout=dropout,
                                        kernel_regularizer=regularizer,
                                        unroll=unroll, # True for GPU usage, false otherwise
                                        #recurrent_dropout
                                        return_sequences=True)(tensor)
        if layerNormalization:
            tensor = keras.layers.LayerNormalization()(tensor)
        tensor = keras.layers.Activation(dense_activation)(tensor)

        # 1D average pooling
        tensor = keras.layers.AveragePooling1D(pool_size=kernel_size,
                                   strides=strides,
                                   padding='same')(tensor)

    # GRU layer (end sequence)
    tensor = keras.layers.GRU(units=n,
                                    activation=GRU_activation,
                                    recurrent_activation='sigmoid',
                                    dropout=dropout,
                                    kernel_regularizer=regularizer,
                                    unroll=unroll, # True for GPU usage, false otherwise
                                    #recurrent_dropout
                                    return_sequences=False)(tensor)

    for i,n in enumerate(dense_layers):
        # Dense layer
        tensor = keras.layers.Dense(units=n,
                                    activation=dense_activation,
                                    use_bias=True,
                                    kernel_regularizer=regularizer)(tensor)
        if batchNormalization:
            tensor = keras.layers.BatchNormalization()(tensor)
        tensor = keras.layers.Activation(dense_activation)(tensor)

    # Dense layer output softmax
    output_tensor = keras.layers.Dense(units=n_classes,
                                    activation='softmax',
                                    use_bias=True)(tensor)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=lrate,
                                amsgrad=False,
                                clipnorm=1e-2)

    model.compile(optimizer=opt,
                        metrics=metrics,
                        loss=loss)

    return model

 
########################################################################### 


def create_MHA(
                                  loss,
                                  metrics,
                                  string_len=3934,
                                  MHA_layers=[50],
                                  nfilters=10,
                                  kernel_size=2, # kernel size and strides should be the same here
                                  strides=2,
                                  dense_layers=[100],
                                  dropout=0.1,
                                  lambda_l2=None,  # None or a float
                                  lrate=1e-4, 
                                  n_classes=46,
                                  conv_activation='relu',
                                  dense_activation='relu',
                                  batchNormalization=True,
                                  layerNormalization=True,
                                  n_tokens=100,
                                  n_embeddings=20, # number of embeddings should be smaller than number of unique tokens (n_tokens)
):

    
    if lambda_l2 is not None:
        # assume a float
        regularizer = tf.keras.regularizers.l2(lambda_l2)
    else:
        regularizer = None

    tensor = keras.Input(shape=(string_len,))

    # dim (batch, len)

    input_tensor = tensor

    # Embedding layer
    tensor = keras.layers.Embedding(n_tokens,
                              n_embeddings,
                              input_length=string_len)(tensor)

    # dim (batch, len, n_embeddings)

    # Positional Encoding layer
    #tensor = PositionalEncoding(max_steps=n_tokens,
    tensor = PositionalEncoding(max_steps=string_len,
                                max_dims=n_embeddings)(tensor)

    # Conv1D layer
    tensor = keras.layers.Conv1D(filters=nfilters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding='same',
                                 kernel_regularizer=regularizer,
                                 activation=conv_activation)(tensor)


    for i,n in enumerate(MHA_layers):
        # Multi-Headed Attention layer (n should be small relative to the vectors stored in the previous tensor)
        tensor = keras.layers.MultiHeadAttention(num_heads=8,
                                             dropout=dropout,
                                             kernel_regularizer=regularizer,
                                             key_dim=n)(tensor, tensor)
        if layerNormalization:
            tensor = keras.layers.LayerNormalization()(tensor)
        tensor = keras.layers.Activation(dense_activation)(tensor)

        # 1D average pooling
        tensor = keras.layers.AveragePooling1D(pool_size=kernel_size,
                                   strides=strides,
                                   padding='same')(tensor)

    # Global max pooling
    tensor = keras.layers.GlobalMaxPooling1D()(tensor)

    for i,n in enumerate(dense_layers):
        # Dense layer
        tensor = keras.layers.Dense(units=n,
                                    activation=dense_activation,
                                    use_bias=True,
                                    kernel_regularizer=regularizer)(tensor)
        if batchNormalization:
            tensor = keras.layers.BatchNormalization()(tensor)
        tensor = keras.layers.Activation(dense_activation)(tensor)

    # Dense layer output softmax
    output_tensor = keras.layers.Dense(units=n_classes,
                                    activation='softmax',
                                    use_bias=True)(tensor)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=lrate,
                                amsgrad=False,
                                clipnorm=1e-2)

    model.compile(optimizer=opt,
                        metrics=metrics,
                        loss=loss)

    return model


