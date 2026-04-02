# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

import pickle
import argparse
import os
import sys


# Location for libraries (you will likely just use './')
tf_tools = "distribute/"

sys.path.append(tf_tools)


# Helper Files
from job_control import *
from hw2 import exp_type_to_hyperparameters
from hw2 import augment_args
from hw2 import generate_fname


def aggregate_stats(args:argparse.ArgumentParser, results_keys:[str]=[], exp_keys:[str]=[]):
    '''
    Aggregate all of the run/results data together for a single set of runs.

    Uses the JobIterator as configured by args.exp_type to enumerate all of the
    result pickle files.  These are loaded sequentially, and the details of each
    run (configuration and results) are then placed into a single dataframe.

    :param args: The arguments from the command line
    :param results_keys: List of strings that are the keys to extract from
                 the results dictionary and store in the dataframe.  Note that
                 these keys must access scalar values!
    :param exp_keys: List of strings that are the keys to extract from
                 the args object and store in the dataframe.  Note that these
                 keys must access scalar values!

    :return: A dataframe that contains one row for each experiment (except for
                 those that have missing pkl file)
    '''

    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be loading
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    # output dataframe list
    dfs = []

    # Iterate over every experiment in the Cartesian product
    for i in range(ji.get_njobs()):
        # Dictionary that describes the DataFrame row
        d = {}

        # Set the exp index to this iteration
        args.exp_index = i

        # Use the exp_index and exp_type to override
        #  specified arguments.  It is a little weird to
        #  do it this way (having two identical JobIterators),
        #  but it does work
        params_str = augment_args(args)

        # Log the experiment conditions
        for k in exp_keys:
            d[k] = getattr(args, k, None)

        # Compute input file name base
        fbase = generate_fname(args, params_str)
    
        # Full input pickle file name
        fname_in = "%s_results.pkl"%(fbase)

        print("File name:", fname_in)

        d['fbase'] = fbase
        d['fname'] = fname_in

        # If the file exists, then load it and create the row
        if os.path.exists(fname_in):
            with open(fname_in, "rb") as fp:
                # Load the data
                results = pickle.load(fp)

                # Log the performance metrics
                for k in results_keys:
                    d[k] = results[k]

                # New row to the dataframe
                dfs.append(pd.DataFrame([d]))
        else:
            # TODO: should we include a blank row?
            print("WARNING -- file does not exist: %s"%fname_in)

    # Combine all of the rows together into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    return df

