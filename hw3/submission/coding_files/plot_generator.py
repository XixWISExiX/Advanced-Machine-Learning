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
from core50 import *
from sklearn.metrics import confusion_matrix
import seaborn as sns



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
        name_shallow = "results/image_Net_ShallowNet_Shallow_Csize_6_Cfilters_15_Pool_4_Pad_valid_hidden_300_drop_0.500_sdrop_0.200_L2_0.000100_LR_0.001000_ntrain_03_rot_0%s_results.pkl"%(str(i))

        name_deep = "results/image_Net_DeepNet_Deep_Csize_5_4_3_Cfilters_10_30_90_Pool_2_2_2_Pad_same_hidden_150_100_50_drop_0.500_sdrop_0.200_L2_0.000100_LR_0.001000_ntrain_03_rot_0%s_results.pkl"%(str(i))

        with open(name_shallow, 'rb') as f:
            shallow_results.append(pickle.load(f))

        with open(name_deep, 'rb') as f:
            deep_results.append(pickle.load(f))

        # Get shallow and deep models
        model_name_shallow = "results/image_Net_ShallowNet_Shallow_Csize_6_Cfilters_15_Pool_4_Pad_valid_hidden_300_drop_0.500_sdrop_0.200_L2_0.000100_LR_0.001000_ntrain_03_rot_0%s_model.keras"%(str(i))

        model_name_deep = "results/image_Net_DeepNet_Deep_Csize_5_4_3_Cfilters_10_30_90_Pool_2_2_2_Pad_same_hidden_150_100_50_drop_0.500_sdrop_0.200_L2_0.000100_LR_0.001000_ntrain_03_rot_0%s_model.keras"%(str(i))

        shallow_models.append(keras.models.load_model(model_name_shallow))

        deep_models.append(keras.models.load_model(model_name_deep))

    # Map the results and models to one location for ease of use
    mapper = {}
    mapper['shallow_results'] = shallow_results
    mapper['deep_results'] = deep_results
    mapper['shallow_models'] = shallow_models
    mapper['deep_models'] = deep_models

    return mapper



def load_in_testing_data() -> list[object]:
    # Open meta data file to pull out the number of objects
    fname = '/home/fagg/datasets/core50/datasets_by_fold_4_objects/meta.csv'
    print(fname)
    df = pd.read_csv(fname)

    nobjects = df['nclasses'].values[0]
    nfolds = df['nfolds'].values[0]

    print('Number of Objects:', nobjects)
    print('Number of Folds:', nfolds)

    # Test create object-based rotations
    core = Core50('/home/fagg/datasets/core50/core50_df.pkl')

    # Load the fold-wise data sets
    testing_rotation_indices = []
    ds_folds = []
    testing_folds = []
    for i in range(5):
        name = 'fold_%d'%i

        testing_rotation_indices.append((nfolds - 1 + i) % nfolds)
        ds = tf.data.Dataset.load('/home/fagg/datasets/core50/datasets_by_fold_4_objects/%s'%(name))
        ds_folds.append(ds)

    # Store each fold
    ds_testing_R0 = ds_folds[testing_rotation_indices[0]]
    ds_testing_R1 = ds_folds[testing_rotation_indices[1]]
    ds_testing_R2 = ds_folds[testing_rotation_indices[2]]
    ds_testing_R3 = ds_folds[testing_rotation_indices[3]]
    ds_testing_R4 = ds_folds[testing_rotation_indices[4]]
    return [ds_testing_R0, ds_testing_R1, ds_testing_R2, ds_testing_R3, ds_testing_R4]



def generate_image(image:list[int], label:int, status:bool, model:str):
    filename = f"./results/fig3_label_{label}_{status}_{model}.png"
    image = tf.squeeze(image)
    image_numpy = image.numpy() 
    # save image
    plt.imsave(filename, image_numpy)



def generate_figure_3(mapper:dict[str, list[object]], ds_testing_array:list[tf.data.Dataset]):
    print('Figure 3')

    bool_array = [0] * 8
    for i in range(5):
        deep_model = mapper['deep_models'][i]
        shallow_model = mapper['shallow_models'][i]
        print("shallow images")
        for image, label in ds_testing_array[i]:  
            if 0 not in bool_array: break # just stop making images
            image = tf.expand_dims(image, axis=0)
            pred = shallow_model.predict(image)
            label = label.numpy()

            # Given an image, make a prediction.
            # We want 8 cases, TP and FP for each of the 4 classes.
            # Shallow
            if label == 0 and np.argmax(pred[0]) == 0 and bool_array[0] != 1:
                print(pred[0], label)
                generate_image(image, label, True, 'shallow')
                bool_array[0] = 1
            elif label == 1 and np.argmax(pred[0]) == 1 and bool_array[1] != 1:
                print(pred[0], label)
                generate_image(image, label, True, 'shallow')
                bool_array[1] = 1
            elif label == 2 and np.argmax(pred[0]) == 2 and bool_array[2] != 1:
                print(pred[0], label)
                generate_image(image, label, True, 'shallow')
                bool_array[2] = 1
            elif label == 3 and np.argmax(pred[0]) == 3 and bool_array[3] != 1:
                print(pred[0], label)
                generate_image(image, label, True, 'shallow')
                bool_array[3] = 1
            elif label == 0 and np.argmax(pred[0]) != 0 and bool_array[4] != 1:
                print(pred[0], label)
                generate_image(image, label, False, 'shallow')
                bool_array[4] = 1
            elif label == 1 and np.argmax(pred[0]) != 1 and bool_array[5] != 1:
                print(pred[0], label)
                generate_image(image, label, False, 'shallow')
                bool_array[5] = 1
            elif label == 2 and np.argmax(pred[0]) != 2 and bool_array[6] != 1:
                print(pred[0], label)
                generate_image(image, label, False, 'shallow')
                bool_array[6] = 1
            elif label == 3 and np.argmax(pred[0]) != 3 and bool_array[7] != 1:
                print(pred[0], label)
                generate_image(image, label, False, 'shallow')
                bool_array[7] = 1

        print("deep images")
        bool_array = [0] * 8
        for image, label in ds_testing_array[i]:  
            if 0 not in bool_array: return # just stop making images
            image = tf.expand_dims(image, axis=0)
            pred = deep_model.predict(image)
            label = label.numpy()
            # Deep
            if label == 0 and np.argmax(pred[0]) == 0 and bool_array[0] != 1:
                print(pred[0], label)
                generate_image(image, label, True, 'deep')
                bool_array[0] = 1
            elif label == 1 and np.argmax(pred[0]) == 1 and bool_array[1] != 1:
                print(pred[0], label)
                generate_image(image, label, True, 'deep')
                bool_array[1] = 1
            elif label == 2 and np.argmax(pred[0]) == 2 and bool_array[2] != 1:
                print(pred[0], label)
                generate_image(image, label, True, 'deep')
                bool_array[2] = 1
            elif label == 3 and np.argmax(pred[0]) == 3 and bool_array[3] != 1:
                print(pred[0], label)
                generate_image(image, label, True, 'deep')
                bool_array[3] = 1
            elif label == 0 and np.argmax(pred[0]) != 0 and bool_array[4] != 1:
                print(pred[0], label)
                generate_image(image, label, False, 'deep')
                bool_array[4] = 1
            elif label == 1 and np.argmax(pred[0]) != 1 and bool_array[5] != 1:
                print(pred[0], label)
                generate_image(image, label, False, 'deep')
                bool_array[5] = 1
            elif label == 2 and np.argmax(pred[0]) != 2 and bool_array[6] != 1:
                print(pred[0], label)
                generate_image(image, label, False, 'deep')
                bool_array[6] = 1
            elif label == 3 and np.argmax(pred[0]) != 3 and bool_array[7] != 1:
                print(pred[0], label)
                generate_image(image, label, False, 'deep')
                bool_array[7] = 1
            





def generate_figure_4(mapper:dict[str, list[object]]):
    print('Figure 4')
    total_labels_shallow = np.array([])
    total_labels_deep = np.array([])
    total_pred_shallow = []
    total_pred_deep = []

    # Go through each model (deep and shallow) and get labels, and results
    for i in range(5):
        print(mapper['shallow_results'][i]['label_testing'])
        total_labels_shallow = np.append(total_labels_shallow, mapper['shallow_results'][i]['label_testing'])
        total_labels_deep = np.append(total_labels_deep, mapper['deep_results'][i]['label_testing'])

        for j in range(len(mapper['shallow_results'][i]['predict_testing'])):
            total_pred_shallow.append(np.argmax(mapper['shallow_results'][i]['predict_testing'][j]))
            total_pred_deep.append(np.argmax(mapper['deep_results'][i]['predict_testing'][j]))

    print(len(total_labels_shallow), len(total_pred_shallow))

    # Create a confusion matrix of the shallow preductions
    fig, ax = plt.subplots()
    cm = confusion_matrix(total_labels_shallow, total_pred_shallow)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3], yticklabels=[0,1,2,3], ax=ax)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Shallow Network Confusion Matrix")

    save_location = './results/fig4a.png'
    fig.savefig(save_location)

    #--------------------------------------------------------------

    # Create a confusion matrix of the deep preductions
    fig, ax = plt.subplots()
    cm = confusion_matrix(total_labels_deep, total_pred_deep)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3], yticklabels=[0,1,2,3], ax=ax)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Deep Network Confusion Matrix")

    save_location = './results/fig4b.png'
    fig.savefig(save_location)


        

def generate_figure_5(mapper:dict[str, list[object]]):
    print('Figure 5')
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

    save_location = './results/fig5.png'
    fig.savefig(save_location)



#################################################################
if __name__ == "__main__":
    mapper = get_file_content()
    generate_figure_5(mapper)
    ds_testing_array = load_in_testing_data()
    print(ds_testing_array)
    generate_figure_3(mapper, ds_testing_array)
    print('-------')
    generate_figure_4(mapper)
    
