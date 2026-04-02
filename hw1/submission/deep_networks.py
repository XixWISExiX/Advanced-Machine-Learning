import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, InputLayer, Dense
from keras import Input, Model
from keras.utils import plot_model


def deep_network_basic(n_inputs:int, n_hidden:[int], n_output:int, activation:str='relu', lrate:float=0.001, activation_out:str='sigmoid' , metrics:[keras.metrics]=[keras.metrics.RootMeanSquaredError()], min_delta:float=0.001):
    '''
    Construct a network with one hidden layer
    - Adam optimizer
    - MSE loss
    
    :param n_inputs: Number of input dimensions
    :param n_hidden: Number of units in the hidden layer
    :param n_output: Number of ouptut dimensions
    :param activation: Activation function to be used for hidden units
    :param lrate: Learning rate for Adam Optimizer
    :param activation_out: Activation function to be used for output units
    :param metrics: Metrics to evaluate model
    :param min_delta: Minimum rate of change over an epoch has to be greater than for the model to keep training
    '''

    model = Sequential()
    model.add(InputLayer(shape=(n_inputs,)))

    # Hidden layers
    for i, n in enumerate(n_hidden):
        model.add(Dense(n, use_bias=True, activation=activation, name='hidden_%d'%i))

    # Output layer
    model.add(Dense(n_output, use_bias=True, activation=activation_out, name='output'))
    
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    
    # Bind the optimizer and the loss function to the model
    model.compile(loss='mse', metrics=metrics, optimizer=opt)

    # Stop training early if we stop making progress
    early_stopping_cb = keras.callbacks.EarlyStopping(restore_best_weights=True,
                                                      min_delta=min_delta,
                                                      monitor='loss')
    
    return model
