import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import argparse
import pickle
import keras
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.colors as colors

from mesonet_support import *
from parser import *
import random
from scipy.stats import skew, kurtosis


def get_test_set(fold:int) -> tf.data.Dataset:
    _, _, _, _, test_dataset, test_labels = get_mesonet_folds(rotation=fold, keep_time=False)
    return test_dataset, test_labels



# This function loads all 8 models which we are going to analyze.
# (This includes the rotations)
def get_file_content() -> dict[str, list[object]]:
    print('Get file content')

    # Initiallize arrays to gather content
    deep_results = []
    deep_models = []
    
    for i in range(8):
        # Get deep model results

        #name_deep = "results/pnn_PNN-DEBUG-1PNN-DEBUG-1__valid_Pad_150_100_50_hidden_LR_0.001000_ntrain_03_rot_0%s_results.pkl"%(str(i))
        #name_deep = "results/pnn_PNN-TEST-1PNN-TEST-1__valid_Pad_256_128_64_32_16_8_hidden_LR_0.000100_ntrain_03_rot_0%s_results.pkl"%(str(i))
        name_deep = "results/pnn_PNN-TEST-3PNN-TEST-3__valid_Pad_64_32_16_8_hidden_LR_0.000001_ntrain_03_rot_0%s_results.pkl"%(str(i))

        with open(name_deep, 'rb') as f:
            deep_results.append(pickle.load(f))

       # # Get deep models
       # model_name_deep = "results/pnn_PNN-DEBUG-1PNN-DEBUG-1__valid_Pad_150_100_50_hidden_LR_0.001000_ntrain_03_rot_0%s_model.keras"%(str(i))

       # deep_models.append(keras.models.load_model(model_name_deep, safe_mode=False))

    # Map the results and models to one location for ease of use
    mapper = {}
    mapper['deep_results'] = deep_results
    #mapper['deep_models'] = deep_models

    return mapper



# Figure 2
# NOTE ONE SINGULAR TEST DATA SET
def generate_distribution_variances(mapper:dict[str, list[object]]):
    print('Figure 2')
    fold = 3

    test_dataset, _ = get_test_set(fold)

    obs = mapper['deep_results'][fold]['label_testing'] # A single test fold
    print(obs.shape)
    #preds = mapper['deep_results'][fold]['predict_testing']
    dists = mapper['deep_results'][fold]['distribution']
    preds = dists.sample(len(obs)) # 1-1 sample to observation predictions
    
    per90list = []
    per75list = []
    per50list = []
    per25list = []
    per10list = []
    for i in range(len(obs)):
        per90, per75, per50, per25, per10 = np.quantile(preds[i], [0.1, 0.25, 0.5, 0.75, 0.9])

        per90list.append(per90)
        per75list.append(per75)
        per50list.append(per50)
        per25list.append(per25)
        per10list.append(per10)
         
#       (43929, 36)       (43929,1)
    plt.plot(test_dataset, 0.000001 + obs, 'c.', alpha=0.3)
    #plt.plot(test_dataset, np.log(0.01 + obs, 'c.', alpha=0.3)
    plt.xlim(800, 900) # Range to look at
    plt.yscale('log')
    #plt.ylim(0, 0.5) # Range to look at


    #plt.plot(dists.quantile(0.9), color='purple', linestyle='--', label='90th Percentile')
    plt.plot(per90list, color='purple', linestyle='--', label='90th Percentile')
    plt.plot(per75list, color='green', linestyle='--', label='75th Percentile')
    plt.plot(per50list, color='blue', linestyle='-', label='50th Percentile')
    plt.plot(per25list, color='red', linestyle='--', label='25th Percentile')
    plt.plot(per10list, color='orange', linestyle='--', label='10th Percentile')


    plt.title(f'Distribution Variance Plot for Fold {fold}')
    plt.xlabel('Time Period (t)')
    plt.ylabel('Precipitation (in)')
    plt.legend()
    plt.savefig('plot_results/fig2.png')
    plt.close()
    


# Figure 3
def stats_metrics_scatter_plot(mapper:dict[str, list[object]]):
    print('Figure 3')
    means = np.array([])
    scales = np.array([])
    skewnesses = np.array([])
    tailweights = np.array([])
    total_obs = np.array([])
    for i in range(8):
        obs = mapper['deep_results'][i]['label_testing']
        #sample_amount = len(obs)//100
        #total_obs = np.concatenate([total_obs, obs.flatten()[:sample_amount]])
        total_obs = np.concatenate([total_obs, obs.flatten()])
        print("total_obs shape:", total_obs.shape)

        dists = mapper['deep_results'][i]['distribution']
        samples = dists.sample(len(obs)) # 1-1 sample to observation predictions
        #samples = dists.sample(sample_amount)
        print("samples shape:", samples.shape)

        means = np.concatenate([means, samples[:,0]])
        scales = np.concatenate([scales, samples[:,1]])
        skewnesses = np.concatenate([skewnesses , samples[:,2]])
        tailweights = np.concatenate([tailweights, samples[:,3]])

    # Create scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Figure 3a
    print(obs.shape, "vs", means.shape)
    axes[0, 0].scatter(total_obs, means, alpha=0.1, color='blue')
    axes[0, 0].set_xlabel("Observed Precipitation (in)")
    axes[0, 0].set_ylabel("Predicted Mean")
    axes[0, 0].set_title("Figure 3a: Predicted Mean vs Observed Precipitation")

    # Figure 3b
    axes[0, 1].scatter(total_obs, scales, alpha=0.1, color='red')
    axes[0, 1].set_xlabel("Observed Precipitation (in)")
    axes[0, 1].set_ylabel("Predicted Scale")
    axes[0, 1].set_title("Figure 3b: Predicted Scale vs Observed Precipitation")

    # Figure 3c
    axes[1, 0].scatter(total_obs, skewnesses, alpha=0.1, color='green')
    axes[1, 0].set_xlabel("Observed Precipitation (in)")
    axes[1, 0].set_ylabel("Predicted Skewness")
    axes[1, 0].set_title("Figure 3c: Predicted Skewness vs Observed Precipitation")

    # Figure 3d
    axes[1, 1].scatter(total_obs, tailweights, alpha=0.1, color='purple')
    axes[1, 1].set_xlabel("Observed Precipitation (in)")
    axes[1, 1].set_ylabel("Predicted Tailweight")
    axes[1, 1].set_title("Figure 3d: Predicted Tailweight vs Observed Precipitation")

    plt.tight_layout()
    plt.savefig('plot_results/fig3.png')
    plt.close()



# Figure 4
def barplot_midpoint_comparision(mapper:dict[str, list[object]]):
    print('Figure 4')
    means = []
    medians = []
    zeros = []
    rotations = []
    for i in range(8):
        obs = mapper['deep_results'][i]['label_testing'] # A single test fold
        dists = mapper['deep_results'][i]['distribution']
        x_vec = dists.sample(len(obs))[:,0] # 1-1 sample to observation predictions

#        MAD_mean = np.sum(x_vec - np.mean(x_vec)) / len(x_vec)
#        MAD_median = np.sum(x_vec - np.median(x_vec)) / len(x_vec)
#        MAD_zero = np.sum(x_vec) / len(x_vec)

        #MAD_mean = np.abs(np.sum(obs - np.mean(x_vec))) / len(obs)
        MAD_mean = np.mean(np.abs(obs - np.mean(x_vec)))
        MAD_median = np.mean(np.abs(obs - np.median(x_vec)))
        MAD_zero = np.mean(np.abs(obs))

        means.append(MAD_mean)
        medians.append(MAD_median)
        zeros.append(MAD_zero)
        rotations.append(f"R{i}")
        print(MAD_mean, MAD_median, MAD_zero)

        #print(mapper['deep_results'][i]['MAD_median_test']) # Medians are 0


    # Bar width and positions
    bar_width = 0.25
    x = np.arange(len(rotations)) * 1.5

    # Plot
    plt.bar(x - bar_width, means, width=bar_width, color="blue", label='MAD Mean')
    plt.bar(x, medians, width=bar_width, color="red", label='MAD Median')
    plt.bar(x + bar_width, zeros, width=bar_width, color="black", label='MAD Zero')

    # Labels
    plt.xlabel('Rotation')
    plt.ylabel('MAD Value')
    plt.title('MAD Metrics per Rotation')
    plt.xticks(x, rotations)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_results/fig4.png')
    plt.close()



#################################################################
if __name__ == "__main__":
    mapper = get_file_content() # NOTE Needed to load models & model info

    barplot_midpoint_comparision(mapper) # Figure 4

    stats_metrics_scatter_plot(mapper) # Figure 3

    generate_distribution_variances(mapper) # Figure 2


