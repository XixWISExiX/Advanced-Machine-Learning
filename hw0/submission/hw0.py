'''
Deep Learning Template

Command line version

Template from: Andrew H. Fagg (andrewhfagg@gmail.com)
'''

import sys
import argparse
import copy
import pickle
import random
import numpy as np
import os
import wandb
import socket

import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, InputLayer, Dense
from keras import Input, Model
from keras.utils import plot_model

import matplotlib.pyplot as plt
from matplotlib import colors

#################################################################
# Default plotting parameters
FONTSIZE = 18
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

#################################################################


def build_model(n_inputs:int, n_hidden:[int], n_output:int, hidden_activation:str='elu', output_activation:str='sigmoid', lrate:float=0.001)-> Sequential:
    '''
    Construct a network with one hidden layer
    - Adam optimizer
    - MSE loss
    
    :param n_inputs: Number of input dimensions
    :param n_hidden: Number of units in the hidden layer
    :param n_output: Number of ouptut dimensions
    :param activation: Activation function to be used for hidden and output units
    :param lrate: Learning rate for Adam Optimizer
    '''
    model = Sequential()
    model.add(InputLayer(shape=(n_inputs,)))

    # Hidden layers
    for i, n in enumerate(n_hidden):
        model.add(Dense(n, use_bias=True, activation=hidden_activation, name='hidden_%d'%i))

    # Output layer
    model.add(Dense(n_output, use_bias=True, activation=output_activation, name='output'))
    
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    
    # Bind the optimizer and the loss function to the model
    model.compile(loss='mse', optimizer=opt)
    
    # Generate an ASCII representation of the architecture
    if args.verbose >= 1:
        print(model.summary())

    return model


def args2string(args:argparse.ArgumentParser)->str:
    '''
    Translate the current set of arguments
    
    :param args: Command line arguments
    '''
    return "%s_%02d_hidden_%s"%(args.label, args.exp, '_'.join([str(i) for i in args.hidden]))
    
    
########################################################
def execute_exp(args:argparse.ArgumentParser):
    '''
    Execute a single instance of an experiment.  The details are specified in the args object
    
    :param args: Command line arguments
    '''

    # Log into weights and biases (currently not here due to obvious reasons...)
    # wandb.login(key="")
                      

    ##############################
    # Run the experiment

    # Create training set: hw0
    # Read inputs for hw0
    with open("/home/fagg/datasets/hw0/hw0_dataset.pkl", "rb") as fp:
        dic = pickle.load(fp)

    # Convert inputs and outputs from -1/1 to 0/1 (binary standard form).
    ins = np.array((dic['ins'] + 1)//2)
    outs = np.array((dic['outs'] + 1)//2)
    #####

    # Create the model
    model = build_model(ins.shape[1], args.hidden, outs.shape[1], hidden_activation=args.hidden_activation, output_activation=args.output_activation, lrate=args.lrate)

    # Callbacks

    # Stop training early if we stop making progress
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=1000,
                                                      restore_best_weights=True,
                                                      min_delta=0.001,
                                                      monitor='loss')
    

    # Describe arguments
    argstring = args2string(args)
    print("EXPERIMENT: %s"%argstring)

    # Output pickle file
    fname_output = "results/hw0_results_%s.pkl"%(argstring)

    # Does this file already exist?
    if os.path.exists(fname_output):
        print("File %s already exists."%fname_output)
        return

    # Plot the model
    if args.render:
        plot_model(model, to_file='results/%s_model_plot.png'%argstring, show_shapes=True, show_layer_names=True)

    
    # Only execute if we are 'going'
    if not args.nogo:

        wandb.init(
            project=args.project,
            name='%s_E%d'%(args.label, args.exp),
            notes=argstring,
            config=vars(args)
        )

        if args.render:
            wandb.log({'architecture': wandb.Image('results/%s_model_plot.png'%argstring)})

            # Log hostname
            wandb.log({'hostname': socket.gethostname()})

        # WandB callback
        wandb_metrics_cb = wandb.keras.WandbMetricsLogger()


        # Training
        print("Training...")
        
        history = model.fit(x=ins,
                            y=outs,
			    epochs=args.epochs,
                            verbose=args.verbose>=2,
                            callbacks=[early_stopping_cb, wandb_metrics_cb]
            )
        
        print("Done Training")
        

        # Record and save model results
        results = {}
        res = model.evaluate(ins, outs)

        results['mse_train'] = res
        results['ins'] = ins
        results['outs'] = outs

        # Log to WandB
        wandb.log(results)

        # Generate learning curve for wandb
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')

        fig.subplots_adjust(left=0.1, right=1.0, bottom=0, top=1.0)

        wandb.log({'mse_plot': wandb.Image(fig)})

        plt.close(fig)

        # Have the model make predictions
        pred = model.predict(ins)

        # Get different error metrics from the model
        abs_errors = np.abs(pred - outs)
        max_error = np.max(abs_errors)
        sum_error = np.sum(abs_errors)
        num_big_errors = len(abs_errors[abs_errors > 0.1])

        wandb.log({"max_error":max_error, "sum_error":sum_error, "num_big_errors":num_big_errors})

        # Save the training history
        with open(fname_output, "wb") as fp:
            pickle.dump(history.history, fp)
            pickle.dump(args, fp)
            pickle.dump(results, fp)
            pickle.dump(abs_errors, fp)
            pickle.dump(max_error, fp)
            pickle.dump(sum_error, fp)

    wandb.finish()

def create_parser()->argparse.ArgumentParser:
    '''
    Create a command line parser for the hw0 experiment
    '''
    parser = argparse.ArgumentParser(description='hw0 Learner')

    parser.add_argument('--exp', type=int, default=0, help='Experiment number')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden', nargs='+', type=int, default=[10,5], help='Number of hidden units')
    parser.add_argument('--lrate', type=float, default=0.001, help='The learning rate for loss optimization')
    parser.add_argument('--hidden_activation', type=str, default='sigmoid', help='The activation function for the hidden layers')
    parser.add_argument('--output_activation', type=str, default='sigmoid', help='The activation function for the output layer')
                    
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--project', type=str, default='test', help='WandB project name')
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files");
    parser.add_argument('--render', action='store_true', help='Render Model')

    return parser

'''
This next bit of code is executed only if this python file itself is executed
(if it is imported into another file, then the code below is not executed)
'''
if __name__ == "__main__":
    # Parse the command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Do the work
    execute_exp(args)
