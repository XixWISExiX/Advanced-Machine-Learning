'''
Advanced Machine Learning, 2025

Author: Andrew H. Fagg (andrewhfagg@gmail.com)

Updates for using GPUs
- Batch file:
#SBATCH --gres=gpu:1

#SBATCH --partition=SOME PARTITION THAT CONTAINS A GPU
or
#SBATCH --partition=disc_dual_a100_students

#SBATCH --cpus-per-task=64       # This will vary depending on the node you are on


- Command line options to include
--cache ""                                     (use RAM to cache the datasets)
OR
--cache $LSCRATCH                              (use lscratch to cache the datasets to local fast disk)


--batch 4096                                   (this parameter is per GPU)
--gpu
--precache datasets_by_fold_4_objects          (use a 4-object pre-constructed dataset)


Notes: 
- batch is now a parameter per GPU.  If there are two GPUs, then this number is doubled internally.
   Note that you must do other things to make use of more than one GPU
- batch of 4096 works on the a100 and h100 GPUs (probably the L40S/ada GPUs)
- The precached dataset is a serialized copy of a set of TF.Datasets (located on slow spinning disk).  
Each directory contains all of the images for a single data fold within a couple of files.  Loading 
these files is *a lot* less expensive than having to load the individual images and preprocess them 
at the beginning of a run.
- The cache is used to to store the loaded datasets into RAM or fast, local SSD so the data can 
be fetched quickly for each training epoch
'''

import sys
import argparse
import pickle
import pandas as pd
import wandb
import socket
import numpy as np
from scipy.stats import skew, kurtosis

from keras.utils import plot_model

#############
# REMOVE THESE LINES
tf_tools = "../support/"
sys.path.append(tf_tools)
#############

# Provided
from job_control import *
from mesonet_support import *
from parser import *

# You need to provide this yourself
from pnn_classifier import *
from MAD_metric import *

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

#################################################################

def exp_type_to_hyperparameters(args:argparse.ArgumentParser)->dict:
    '''
    Translate the exp_type into a hyperparameter set

    This is trivial right now

    :param args: ArgumentParser

    :return: Hyperparameter set (in dictionary form)
    '''
    if args.exp_type is None:
        p = {'rotation': range(8)}
    elif args.exp_type == 'L2':
        p = {'rotation': range(8), 'L2_regularization': [1.0, 10.0]}
    else:
        assert False, "Unrecognized exp_type (%s)"%args.exp_type

    return p


#################################################################
def check_args(args:argparse.ArgumentParser):
    '''
    Check that the input arguments are rational

    '''
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-2)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.dropout is None or (args.dropout > 0.0 and args.dropout < 1)), "Dropout must be between 0 and 1"
    assert (args.spatial_dropout is None or (args.spatial_dropout > 0.0 and args.dropout < 1)), "Spatial dropout must be between 0 and 1"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.L1_regularization is None or (args.L1_regularization > 0.0 and args.L1_regularization < 1)), "L1_regularization must be between 0 and 1"
    assert (args.L2_regularization is None or (args.L2_regularization > 0.0 and args.L2_regularization < 1)), "L2_regularization must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"
    


def augment_args(args:argparse.ArgumentParser)->str:
    '''
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser

    :return: A string representing the selection of parameters to be used in the file name
    '''
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Check index number
    index = args.exp_index
    if index is None:
        return ""
    
    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)
 
    
#################################################################

def generate_fname(args:argparse.ArgumentParser, params_str:str)->str:
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse

    :params_str: String generated by the JobIterator

    :return: Full string for file name
    '''
    # Hidden unit configuration
    hidden_str = '_'.join(str(x) for x in args.hidden)
    
    # Dropout
    if args.dropout is None:
        dropout_str = ''
    else:
        dropout_str = 'drop_%0.3f_'%(args.dropout)
        
    # L1 regularization
    if args.L1_regularization is None:
        regularizer_l1_str = ''
    else:
        regularizer_l1_str = 'L1_%0.6f_'%(args.L1_regularization)

    # L2 regularization
    if args.L2_regularization is None:
        regularizer_l2_str = ''
    else:
        regularizer_l2_str = 'L2_%0.6f_'%(args.L2_regularization)


    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_"%args.label
        
    # Experiment type
    if args.exp_type is None:
        experiment_type_str = ""
    else:
        experiment_type_str = "%s_"%args.exp_type

    # learning rate
    lrate_str = "LR_%0.6f_"%args.lrate
    
    # Put it all together, including #of training folds and the experiment rotation
    return "%s/pnn_%s%s%s_%s_Pad_%s_hidden_%s%s%sntrain_%02d_rot_%02d"%(args.results_path,
                                                                                                          args.label,
                                                                                           experiment_type_str,
                                                                                           label_str,
                                                                                           args.padding,
                                                                                           hidden_str, 
                                                                                           regularizer_l1_str,
                                                                                           regularizer_l2_str,
                                                                                           lrate_str,
                                                                                           args.Ntraining,
                                                                                           args.rotation)


#################################################################

def execute_exp(args:argparse.ArgumentParser=None, multi_gpus:bool=False):
    '''
    Perform the training and evaluation for a single model
    
    :param args: Argparse arguments
    :param multi_gpus: True if there are more than one GPU
    '''

    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    print(args.exp_index)
    
    # Override arguments if we are using exp_index
    args_str = augment_args(args)

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch*multi_gpus

    print('Batch size', args.batch)

    ####################################################
    # Create the TF datasets for training, validation, testing

    if args.verbose >= 3:
        print('Starting data flow')

    # Load data
    ds_train, train_labels, ds_validation, valid_labels, ds_testing, test_labels = get_mesonet_folds(rotation=args.exp_index)

    ####################################################
    # Build the model
    if args.verbose >= 3:
        print('Building network')

    # Create the network
    mad = MeanAbsoluteDeviation()
    inner_model, outer_model = create_pnn_model(n_inputs=36,
                                 n_hidden=args.hidden,
                                 activation=args.activation_dense,
                                 activation_out=args.activation_out,
                                 dropout=args.dropout,
                                 learning_rate=args.lrate,
                                 metrics=[mad])
 
        

    # Report model structure if verbosity is turned on
    if args.verbose >= 1:
        print(inner_model.summary())
        print(outer_model.summary())

    print(args)

    # Output file base and pkl file
    fbase = generate_fname(args, args_str)
    print(fbase)
    fname_out = "%s_results.pkl"%fbase

    # Plot the inner model
    if args.render:
        render_fname = '%s_model_plot.png'%fbase
        plot_model(inner_model, to_file=render_fname, show_shapes=True, show_layer_names=True)

        #data_distribution_fname = 'results/train_data_distribution_plot.png'
        #plt.plot(ds_train, train_labels, 'c.')
        #plt.plot(ds_train, np.mean(ds_train, axis=1), 'r')
        #plt.xlabel('input')
        #plt.ylabel('output')
        #plt.savefig(data_distribution_fname)

    
    # Print training set distribution
    mean_val = np.mean(train_labels)
    scale_val = np.std(train_labels)
    skewness_val = skew(train_labels)
    tailweight_val = kurtosis(train_labels)
    print("=== Training Set ===")
    print(f"Mean: {mean_val}")
    print(f"Scale (std): {scale_val}")
    print(f"Skewness: {skewness_val}")
    print(f"Tailweight (Excess Kurtosis): {tailweight_val}")

    # Print validation set distribution
    mean_val = np.mean(valid_labels)
    scale_val = np.std(valid_labels)
    skewness_val = skew(valid_labels)
    tailweight_val = kurtosis(valid_labels)
    print("=== Validation Set ===")
    print(f"Mean: {mean_val}")
    print(f"Scale (std): {scale_val}")
    print(f"Skewness: {skewness_val}")
    print(f"Tailweight (Excess Kurtosis): {tailweight_val}")


    # See the dimensions of X and y
    print("Train X shape", ds_train.shape)
    print("Train y shape", train_labels.shape)

    # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        print(fbase)
        return

    # Check if output file already exists
    if not args.force and os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists"%fname_out)
        return

    #####
    # Don't steal my key *_*
    # Log into weights and biases
    wandb.login(key="355d903a9d0fe3220c34af3e1bd5a18d54a55a08")

    # Start wandb
    run = wandb.init(project=args.project, name='%s_R%d'%(args.label,args.rotation), notes=fbase, config=vars(args))

    # Log hostname
    wandb.log({'hostname': socket.gethostname()})

    # Log model design image
    if args.render:
        wandb.log({'model architecture': wandb.Image(render_fname)})
        #wandb.log({'data distribution': wandb.Image(data_distribution_fname)})

            
    #####
    # Callbacks
    cbs = []
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta,
                                                      monitor=args.monitor)

    # Bug encountered, where these three lines are needed.
    early_stopping_cb._implements_train_batch_hooks = lambda: False 
    early_stopping_cb._implements_predict_batch_hooks = lambda: False 
    early_stopping_cb._implements_test_batch_hooks = lambda: False

    cbs.append(early_stopping_cb)

    # Weights and Biases
    wandb_metrics_cb = wandb.keras.WandbMetricsLogger()

    # Bug encountered, where these three lines are needed.
    wandb_metrics_cb._implements_train_batch_hooks = lambda: False 
    wandb_metrics_cb._implements_predict_batch_hooks = lambda: False 
    wandb_metrics_cb._implements_test_batch_hooks = lambda: False

    cbs.append(wandb_metrics_cb)

    if args.verbose >= 3:
        print('Fitting model')

    # Learn
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #          Note that if you use this, then you must repeat the training set
    #  validation_steps=None means that ALL validation samples will be used


    # Learn
    history = outer_model.fit(x=ds_train,
                        y=train_labels,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        verbose=args.verbose>=2,
                        validation_data=(ds_validation, valid_labels),
                        validation_steps=None,
                        callbacks=cbs)

    # Done training

    # Generate results data
    results = {}

    # Validation set
    print('#################')
    print('Validation')
    results['args'] = args

    # Test set
    if ds_testing is not None:
        print('#################')
        print('Testing')

        # Get mean, scale, skewness, tailweight for test set
        results['distribution'] = outer_model(ds_testing)

        results['predict_testing'] = outer_model.predict(ds_testing)
        results['label_testing'] = test_labels


    # Training set
    print('#################')
    print('Training')

    results['history'] = history.history

    ## NOTE: may want to add some additional logging of test data performance

    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    with open("%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        outer_model.save("%s_model.keras"%(fbase))

    wandb.finish()

    return outer_model


def check_completeness(args:argparse.ArgumentParser):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser

    '''
    
    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d"%ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)
    
        # Output pickle file name
        fname_out = "%s_results.pkl"%(fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s"%(i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s"%(len(indices),','.join(str(x) for x in indices)))

    
#################################################################
if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    if args.verbose >= 3:
        print('Arguments parsed')

    # Turn off GPU?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')

    # GPU check
    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print('GPUS:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_visible_devices)
    else:
        print('NO GPU')

    if args.check:
        # Just check to see if all experiments have been executed
        check_completeness(args)
    else:
        # Execute the experiment

        # Set number of threads, if it is specified
        if args.cpus_per_task is not None:
            tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task//2)
            tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task//2)

        # Do the work
        execute_exp(args, multi_gpus=n_visible_devices)
        

