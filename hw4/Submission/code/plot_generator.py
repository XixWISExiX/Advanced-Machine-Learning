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

from chesapeake_loader4 import *
from parser import *
import random


def get_test_set(fold:int) -> tf.data.Dataset:
    parser = create_parser()
    args = parser.parse_args()

    ds_training, ds_validation, ds_testing, nobjects = create_datasets(patch_size=args.batch,
    #train_filt='*0',
    #fold=args.exp_index,
    fold=fold,
    repeat_train=args.repeat,
    shuffle_train=args.shuffle,
    prefetch=args.prefetch,
    num_parallel_calls=args.num_parallel_calls)

    return ds_testing



# This function loads all 10 models which we are going to analyze.
# (This includes the rotations)
def get_file_content() -> dict[str, list[object]]:
    print('Get file content')

    # Initiallize arrays to gather content
    shallow_results = []
    deep_results = []

    shallow_models = []
    deep_models = []
    
    for i in range(5):
        # Get shallow and deep model results
        name_shallow = "results/image_Net_ShallowNet_Shallow_Csize_2_Cfilters_600_Pool_2_Pad_same_hidden_300_drop_0.100_sdrop_0.100_L2_0.000100_LR_0.000100_ntrain_03_rot_0%s_results.pkl"%(str(i))

        name_deep = "results/image_Net_Deep_4Net_Deep_4_Csize_2_2_2_2_2_Cfilters_64_256_256_256_64_Pool_2_2_2_2_2_Pad_same_hidden_150_100_50_drop_0.100_sdrop_0.100_LR_0.000100_ntrain_03_rot_0%s_results.pkl"%(str(i))

        with open(name_shallow, 'rb') as f:
            shallow_results.append(pickle.load(f))

        with open(name_deep, 'rb') as f:
            deep_results.append(pickle.load(f))

        # Get shallow and deep models
        model_name_shallow = "results/image_Net_ShallowNet_Shallow_Csize_2_Cfilters_600_Pool_2_Pad_same_hidden_300_drop_0.100_sdrop_0.100_L2_0.000100_LR_0.000100_ntrain_03_rot_0%s_model.keras"%(str(i))

        model_name_deep = "results/image_Net_Deep_4Net_Deep_4_Csize_2_2_2_2_2_Cfilters_64_256_256_256_64_Pool_2_2_2_2_2_Pad_same_hidden_150_100_50_drop_0.100_sdrop_0.100_LR_0.000100_ntrain_03_rot_0%s_model.keras"%(str(i))

        shallow_models.append(keras.models.load_model(model_name_shallow))

        deep_models.append(keras.models.load_model(model_name_deep))

    # Map the results and models to one location for ease of use
    mapper = {}
    mapper['shallow_results'] = shallow_results
    mapper['deep_results'] = deep_results
    mapper['shallow_models'] = shallow_models
    mapper['deep_models'] = deep_models

    return mapper


def generate_image_labels(mapper:dict[str, list[object]]):
    print("Generating Images")

    normal_images = []
    shallow_pred_images = []
    deep_pred_images = []
    true_images = []

    # Loop through each fold
    for fold in range(5):
        ds_testing = get_test_set(fold=fold)
        print("Got testing set, Fold", fold)

        shallow_model = mapper['shallow_models'][fold]
        deep_model = mapper['deep_models'][fold]

        # Choose 2 random images from each fold
        for _ in range(2):
            print('img')

            # Choose a random batch
            randomBatch = None
            randomIdx = random.randint(0,len(list(ds_testing)))
            for i,batch in enumerate(ds_testing):
                if randomIdx == i:
                    randomBatch = batch
            batch_X, batch_y = randomBatch

            # Choose a random image
            random_idx = random.randint(0, batch_X.shape[0]-1)
            random_image = batch_X[random_idx]
            random_image_adj = np.expand_dims(random_image, axis=0)
            random_label = batch_y[random_idx]
            random_label_adj = np.expand_dims(random_label, axis=0)

            batch_shallow_pred = shallow_model.predict(random_image_adj)
            batch_shallow_pred = np.argmax(batch_shallow_pred, axis=-1)
            batch_deep_pred = deep_model.predict(random_image_adj)
            batch_deep_pred = np.argmax(batch_deep_pred, axis=-1)

            # Print dimensions of the image for correctiveness
            print('normal', random_image[:, :, :3].shape)
            print('pred', batch_shallow_pred[0].shape)
            print('true', batch_y[random_idx].shape)

            normal_images.append(random_image[:, :, :3])
            shallow_pred_images.append(batch_shallow_pred[0])
            deep_pred_images.append(batch_deep_pred[0])
            true_images.append(batch_y[random_idx])

    fig, axes = plt.subplots(10, 3, figsize=(15, 30))
    fig.subplots_adjust(hspace=0.4)  # Add space between rows

    for i in range(10):
        # Satellite image (first column)
        axes[i, 0].imshow(normal_images[i])
        axes[i, 0].set_title('Satellite Image')
        axes[i, 0].axis('off')
    
        # True labels (second column)
        axes[i, 1].imshow(true_images[i], cmap='viridis', vmax=6)
        axes[i, 1].set_title('True Labels')
        axes[i, 1].axis('off')
    
        # Predicted labels (third column)
        axes[i, 2].imshow(shallow_pred_images[i], cmap='viridis', vmax=6)
        axes[i, 2].set_title('Predicted Labels')
        axes[i, 2].axis('off')

    save_location = './plot_results/fig5a.png'
    fig.savefig(save_location)

    fig, axes = plt.subplots(10, 3, figsize=(15, 30))
    fig.subplots_adjust(hspace=0.4)  # Add space between rows

    for i in range(10):
        # Satellite image (first column)
        axes[i, 0].imshow(normal_images[i])
        axes[i, 0].set_title('Satellite Image')
        axes[i, 0].axis('off')
    
        # True labels (second column)
        axes[i, 1].imshow(true_images[i], cmap='viridis', vmax=6)
        axes[i, 1].set_title('True Labels')
        axes[i, 1].axis('off')
    
        # Predicted labels (third column)
        axes[i, 2].imshow(deep_pred_images[i], cmap='viridis', vmax=6)
        axes[i, 2].set_title('Predicted Labels')
        axes[i, 2].axis('off')

    save_location = './plot_results/fig5b.png'
    fig.savefig(save_location)




def generate_confusion_matrix(mapper:dict[str, list[object]]):
    print('Generating Confusion Matrix')
    total_labels = []
    total_preds_shallow = []
    total_preds_deep = []

    # Load each fold
    for fold in range(5):
        ds_testing = get_test_set(fold=fold)
        print("Got testing set, Fold", fold)

        shallow_model = mapper['shallow_models'][fold]
        deep_model = mapper['deep_models'][fold]

        # Load each batch and make a prediction
        for i, batch in enumerate(ds_testing):
            print("Batch", i)
            batch_X, batch_y = batch

            batch_shallow_preds = shallow_model.predict(batch_X)
            batch_deep_preds = deep_model.predict(batch_X)

            batch_shallow_preds = np.argmax(batch_shallow_preds, axis=-1)
            batch_deep_preds = np.argmax(batch_deep_preds, axis=-1)

            # Flatten labels and predictions
            total_labels.append(batch_y.numpy().flatten())  
            total_preds_shallow.append(batch_shallow_preds.flatten())  
            total_preds_deep.append(batch_deep_preds.flatten())  


    # Convert lists to numpy arrays for analysis
    total_labels = np.concatenate(total_labels)
    total_preds_shallow = np.concatenate(total_preds_shallow) 
    total_preds_deep = np.concatenate(total_preds_deep) 


    # Create a confusion matrix of the shallow preductions
    fig, ax = plt.subplots(figsize=(10,10))
    cm = confusion_matrix(total_labels, total_preds_shallow, labels=[0,1,2,3,4,5,6])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4,5,6], yticklabels=[0,1,2,3,4,5,6], ax=ax, norm=colors.LogNorm(vmin=1, vmax=cm.max()))  
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Shallow Network Confusion Matrix")
    save_location = './plot_results/fig3a.png'
    fig.savefig(save_location)


    # Create a confusion matrix of the deep preductions
    fig, ax = plt.subplots(figsize=(10,10))
    cm = confusion_matrix(total_labels, total_preds_deep, labels=[0,1,2,3,4,5,6])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4,5,6], yticklabels=[0,1,2,3,4,5,6], ax=ax, norm=colors.LogNorm(vmin=1, vmax=cm.max()))
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Deep Network Confusion Matrix")
    save_location = './plot_results/fig3b.png'
    fig.savefig(save_location)

        

def generate_scatter_plot(mapper:dict[str, list[object]]):
    print('Generating Scatter Plot')
    shallow_testing_acc = []
    deep_testing_acc = []

    # Go through each model and store it's accurracy
    for i in range(5):
        shallow_testing_acc.append(mapper['shallow_results'][i]['predict_testing_eval'][1])
        deep_testing_acc.append(mapper['deep_results'][i]['predict_testing_eval'][1])

    # Create a plot to compare deep and shallow network accuracies
    fig, ax = plt.subplots()

    ax.scatter(shallow_testing_acc, deep_testing_acc, color='red', marker='o', label="Test Set Accuracy")
    ax.plot([0, 1], [0, 1], 'k--', label="Equal Performance Boundary")

    ax.set_xlabel("Shallow Network Accuracy")
    ax.set_ylabel("Deep Network Accuracy")
    ax.set_title("Test Set Accuracy: Shallow vs Deep Networks")
    ax.legend()

    save_location = './plot_results/fig4.png'
    fig.savefig(save_location)



#################################################################
if __name__ == "__main__":
    mapper = get_file_content() # NOTE Needed to load models & model info

    generate_scatter_plot(mapper) # Figure 4

    generate_confusion_matrix(mapper) #Figure 3

    generate_image_labels(mapper) #Figure 5

