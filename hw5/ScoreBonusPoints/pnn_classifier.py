# Credit for most of this code is to Andrew H. Fagg

import argparse
import copy
import pickle
import random
import matplotlib.pyplot as plt 
import numpy as np


import tensorflow as tf

# THIS IS REALLY IMPORTANT
import tf_keras as keras
from tf_keras.models import Sequential
import tensorflow_probability as tfp

# Sub namespaces that are useful later
# Tensorflow Distributions
tfd = tfp.distributions
# Probability Layers 
tfpl = tfp.layers

from tf_keras.layers import Layer, Concatenate
from tf_keras.layers import Dense, BatchNormalization, Dropout
from tf_keras import Input, Model
from matplotlib import colors
from tf_keras.utils import plot_model

from mesonet_support import *

# ~Gaussian density
def gauss(t, mu, sigma):
    return np.exp(-np.square((t-mu)/sigma))

def fully_connected_stack(n_inputs:int, 
                          n_hidden:int, 
                          n_output:int, 
                          activation:str='elu', 
                          activation_out:str=None,
                          lrate:float=0.001, 
                          dropout:float=None, 
                          dropout_input:float=None, 
                          kernel_regularizer_L2:float=None, 
                          kernel_regularizer_L1:float=None):

    '''
    General network building code that creates a stack of Dense layers.
    The output of the stack is a list of output Keras Tensors, each with its own activation function.
    If the output activation function is 'positive', then the output is elu()+1.1
    
    :param n_inputs: Number of input units
    :param n_hidden: List of hidden layer sizes
    :param n_output: List of the number of output units for each output Tensor
    :param activation: Activation function for hidden layers
    :param activation_out: List of the activation functions for each output Tensor
    :param lrate: Learning rate
    :param dropout: Dropout probability for hidden layers
    :param dropout_input: Dropout probability for input layer
    :param kernel_regularizer_L2: L2 regularization param
    :param kernel_regularizer_L1: L1 regularization param
    :return: (input tensor, list of output tensors)
    '''
    
    if dropout is not None:
        print("DENSE: DROPOUT %f"%dropout)

    # L2 or L1 regularization?
    kernel_regularizer = None
    if kernel_regularizer_L2 is not None:
        print("DENSE: L2 Regularization %f"%kernel_regularizer)
        kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)
    elif kernel_regularizer_L1 is not None:
        # Only us L1 if specified *and* L2 is not active
        print("DENSE: L1 Regularization %f"%kernel_regularizer_L1)
        kernel_regularizer=tf.keras.regularizers.l1(kernel_regularizer_L1)

    # Input layer
    input_tensor = tensor = Input(shape=(n_inputs,))
    tensor = BatchNormalization()(tensor)
    
    # Dropout input features?
    if dropout_input is not None:
        tensor = Dropout(rate=dropout_input, name="dropout_input")(tensor)
            
    # Loop over hidden layers
    for i, n in enumerate(n_hidden):             
        tensor = Dense(n, use_bias=True, name="hidden_%02d"%(i), activation=activation,
                 kernel_regularizer=kernel_regularizer)(tensor)
        
        tensor = BatchNormalization()(tensor)
        if dropout is not None:
            tensor = Dropout(rate=dropout, name="dropout_%02d"%(i))(tensor)
    
    # Output layers
    outputs = []
    for i, (n, act) in enumerate(zip(n_output, activation_out)):
        o = Dense(n, use_bias=True, name="output%d"%i, activation=act)(tensor)
        outputs.append(o)

    return input_tensor, outputs
    


def create_pnn_model(n_inputs=1, # Input (t) is 1-dimensional
                     #d_output=1, # Output (y) is 1-dimensional
                     #n_gaussians = 2, # note by using 2, we can toggle the ratio of using the 2 distributions
                     n_hidden=None,
                     #n_output=None,
                     activation='elu',
                     activation_out='linear',
                     dropout=None,
                     learning_rate=0.001,
                     metrics=None
                     ):

    # Example usecase
    #input_tensor, output_tensors  = fully_connected_stack(n_inputs=1, 
    #                                               n_hidden=[100, 50, 20], 
    #                                               n_output=[1],
    #                                               activation='elu',
    #                                               activation_out=['linear'],
    #                                               dropout=None)

    # Number of Gaussians


    #######################
    # NOTE Inner model

    # Compute the number of output units for the inner model
    n_output = SinhArcsinh.num_params()

    # Main model stack
    input_tensor, output_tensors = fully_connected_stack(n_inputs=n_inputs, 
                                                   n_hidden=n_hidden, 
                                                   n_output=[n_output],
                                                   activation=activation,
                                                   activation_out=[activation_out],
                                                   dropout=dropout)



    model_inner = Model(inputs=input_tensor, outputs=output_tensors)
    
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False)

    #######################
    # NOTE Outer model
    # Key: do not mix Keras Tensors between these two models, but
    #  we can use the inner model as a layer in the Outer model

    tensor = input_tensor2 = Input(shape=(n_inputs,))
    output_tensor2 = model_inner(tensor)

    # This layer takes a Keras Tensor as input and returns a TF Probability Distribution
    #  It also handles the constraint that std must be positive
    #  NOTE: this only works right now when using Keras 2

    
    # Prevent vanishing/exploding gradiant issue
    output_tensor2 = tf.nn.softplus(output_tensor2)

    output2 = SinhArcsinh.create_layer()(output_tensor2)

    model_outer = Model(inputs=input_tensor2, outputs=output2)

    # Optimizer
    opt2 = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False)

    # We don't have to use a built-in loss function.  Instead, we use the
    #   one defined above
    print("MAD Debugger (before compile)", metrics)
    print("tensorflow version:", tf.__version__)
    print("keras version:", tf.keras.__version__)
    model_outer.compile(optimizer=opt2, 
                        metrics=metrics, # NOTE MAD metric
                        loss=SinhArcsinh.mad_loss)
 #                       loss=SinhArcsinh.mdn_loss)

    #print(model_outer.summary())

    return model_inner, model_outer


