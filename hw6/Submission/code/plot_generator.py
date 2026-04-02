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

from pfam_loader import *
from parser import *
import random


def get_val_test_set(fold:int) -> tf.data.Dataset:
    parser = create_parser()
    args = parser.parse_args()

    amino_acid_dic = load_rotation(rotation=fold, version='B')
    ds_train, ds_validation, ds_testing = create_tf_datasets(amino_acid_dic,
                                                    #8       #args.batch,
                                                             32,
                                                             args.prefetch,
                                                             args.shuffle,
                                                             #args.repeat,
                                                             False,
                                                             args.cache)

    return ds_validation, ds_testing



# This function loads all 8 models which we are going to analyze.
# (This includes the rotations)
def get_file_content() -> dict[str, list[object]]:
    print('Get file content')

    # Initiallize arrays to gather content
    rnn_results = []
    gru_results = []
    mha_results = []

    rnn_models = []
    gru_models = []
    mha_models = []
    
    for i in range(5):
        # Get model results
        #name_rnn = "results/SimpleRNN___Special_64_32_16_Hidden_100_50_Lr_LR_0.001000__rot_0%s_results.pkl"%(str(i))
        #name_gru = "results/GRU___Special_40_20_10_Hidden_100_50_Lr_LR_0.001000__rot_0%s_results.pkl"%(str(i))
        #name_mha = "results/MHA___Special_40_20_10_Hidden_100_50_Lr_LR_0.001000__rot_0%s_results.pkl"%(str(i))
        name_rnn = "results/SimpleRNN-2___Special_64_32_16_Hidden_128_64_Lr_LR_0.001000__rot_0%s_results.pkl"%(str(i))
        name_gru = "results/GRU-2___Special_64_32_16_Hidden_128_64_Lr_LR_0.001000__rot_0%s_results.pkl"%(str(i))
        #name_mha = "results/MHA-2___Special_64_32_16_Hidden_128_64_Lr_LR_0.000100__rot_0%s_results.pkl"%(str(i))
        name_mha = "results/MHA-4___Special_64_32_16_Hidden_128_64_Lr_LR_0.000100__rot_0%s_results.pkl"%(str(i))

        with open(name_rnn, 'rb') as f:
            rnn_results.append(pickle.load(f))

        with open(name_gru, 'rb') as f:
            gru_results.append(pickle.load(f))

        with open(name_mha, 'rb') as f:
            mha_results.append(pickle.load(f))

        # Get models
        #model_name_rnn = "results/SimpleRNN___Special_64_32_16_Hidden_100_50_Lr_LR_0.001000__rot_0%s_model.keras"%(str(i))
        #model_name_gru = "results/GRU___Special_40_20_10_Hidden_100_50_Lr_LR_0.001000__rot_0%s_model.keras"%(str(i))
        #model_name_mha = "results/MHA___Special_40_20_10_Hidden_100_50_Lr_LR_0.001000__rot_0%s_model.keras"%(str(i))
        model_name_rnn = "results/SimpleRNN-2___Special_64_32_16_Hidden_128_64_Lr_LR_0.001000__rot_0%s_model.keras"%(str(i))
        model_name_gru = "results/GRU-2___Special_64_32_16_Hidden_128_64_Lr_LR_0.001000__rot_0%s_model.keras"%(str(i))
        #model_name_mha = "results/MHA-2___Special_64_32_16_Hidden_128_64_Lr_LR_0.000100__rot_0%s_model.keras"%(str(i))
        model_name_mha = "results/MHA-4___Special_64_32_16_Hidden_128_64_Lr_LR_0.000100__rot_0%s_model.keras"%(str(i))

        rnn_models.append(keras.models.load_model(model_name_rnn, safe_mode=False))
        gru_models.append(keras.models.load_model(model_name_gru, safe_mode=False))
        mha_models.append(keras.models.load_model(model_name_mha, safe_mode=False))

    # Map the results and models to one location for ease of use
    mapper = {}
    mapper['rnn_results'] = rnn_results
    mapper['gru_results'] = gru_results
    mapper['mha_results'] = mha_results

    mapper['rnn_models'] = rnn_models
    mapper['gru_models'] = gru_models
    mapper['mha_models'] = mha_models

    return mapper




# Figure 4
def generate_contingency_table(mapper:dict[str, list[object]]):
    print('Figure 4')

    ds_test_labels = []
    ds_test_preds = []

    # Load each fold
    for fold in range(5):
        _, ds_testing = get_val_test_set(fold=fold)
        print("Got testing set, Fold", fold)

        #model = mapper['rnn_models'][fold]
        #model = mapper['gru_models'][fold]
        model = mapper['mha_models'][fold]

        # Load each batch and make a prediction
        for i, batch in enumerate(ds_testing):
            print("Batch", i)
            batch_X, batch_y = batch

            batch_preds = model.predict(batch_X)

            batch_preds = np.argmax(batch_preds, axis=-1)

            # Flatten labels and predictions
            ds_test_labels.append(batch_y.numpy().flatten())
            ds_test_preds.append(batch_preds.flatten())


    # Convert lists to numpy arrays for analysis
    ds_test_labels = np.concatenate(ds_test_labels)
    ds_test_preds = np.concatenate(ds_test_preds)

    # Create a confusion matrix of the shallow preductions
    fig, ax = plt.subplots(figsize=(40,40))
    cm = confusion_matrix(ds_test_labels, ds_test_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, norm=colors.LogNorm(vmin=1, vmax=cm.max()))
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("MHA Confusion Matrix")
    save_location = './plot_results/fig4.png'
    fig.savefig(save_location)




#################################################################
if __name__ == "__main__":
    mapper = get_file_content() # NOTE Needed to load models & model info

    # or confusion matrix
    generate_contingency_table(mapper) # Figure 4

