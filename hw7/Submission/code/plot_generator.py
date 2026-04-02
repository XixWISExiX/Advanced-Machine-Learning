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
from diffusion_tools import *
import random
from unet_classifier import PositionEncoder, ExpandDimsTwice, TileToImage


def get_validation_set(mapper:dict[str, list[object]]) -> tf.data.Dataset:
    #parser = create_parser()
    #args = parser.parse_args()
    args = mapper['results']['args']

    if args.sine_params:
        # Sine or cosine shape
        beta, alpha, sigma = compute_beta_alpha2(nsteps=args.diffusion_steps,
                beta_start=args.diffusion_beta_start,
                beta_end=args.diffusion_beta_end,
                gamma_start=args.diffusion_gamma_start,
                gamma_end=args.diffusion_gamma_end, cos=False)
    else:
        beta, alpha, sigma = compute_beta_alpha(nsteps=args.diffusion_steps,
                beta_start=args.diffusion_beta_start,
                beta_end=args.diffusion_beta_end,
                gamma_start=args.diffusion_gamma_start,
                gamma_end=args.diffusion_gamma_end)

    #ds_valid = create_single_dataset(
    ds, ds_valid = create_diffusion_dataset(alpha=alpha,
            full_sat=False,
            fold=args.exp_index,
            partition='valid',
            patch_size=args.patch_size,
            repeat=args.repeat,
            shuffle=args.shuffle,
            batch_size=args.batch,
            prefetch=args.prefetch,
            num_parallel_calls=args.num_parallel_calls)
            #time_sampling_exponent=args.time_sampling_exponent)

    return ds_valid, beta, alpha, sigma, args.diffusion_steps




# This function loads all 10 models which we are going to analyze.
# (This includes the rotations)
def get_file_content() -> dict[str, list[object]]:
    print('Get file content')
    # Newest
    fname = "results/image_50-gamer-sine-25650-gamer-sine-256_Csize_3_3_1_Cfilters_96_128_256_Pool_2_2_2_Pad_same_hidden_200_150_100_drop_0.100_sdrop_0.100_LR_0.000100_ntrain_03_rot_00_Diffsteps_50_BetaEnd_0.1000_GammaEnd_0.1000"

    # 0.5 exponent
    #fname = "results/image_50-gamer-sine50-gamer-sine_Csize_3_3_1_Cfilters_64_128_256_Pool_2_2_2_Pad_same_hidden_200_150_100_drop_0.100_sdrop_0.100_LR_0.000100_ntrain_03_rot_00_Diffsteps_50_BetaEnd_0.1000_GammaEnd_0.1000"

    #model_name = "results/image_Net_DeepNet_Deep_Csize_2_2_2_2_2_Cfilters_64_256_256_256_64_Pool_2_2_2_2_2_Pad_same_hidden_150_100_50_drop_0.100_sdrop_0.100_LR_0.000100_ntrain_03_rot_00_Diffsteps_20_BetaEnd_00_GammaEnd_00_model.keras")

    #model = keras.models.load_model(model_name)
    model = keras.models.load_model(fname + '_model.keras',
                  custom_objects={'PositionEncoder': PositionEncoder,
                  'ExpandDims': ExpandDimsTwice,
                  'Tile': TileToImage,
                  #'ExpandDims': keras.src.ops.numpy.ExpandDims,
                  #'Tile': keras.src.ops.numpy.Tile,
                  'mse': 'mse'})
                  #, safe_mode=False)
                  
    with open(fname + '_results.pkl', 'rb') as fp:
        results = pickle.load(fp)



    # Map the results and models to one location for ease of use
    mapper = {}
    mapper['results'] = results
    mapper['model'] = model

    return mapper



def generate_image_noise_over_time(mapper:dict[str, list[object]]):
    print('Generate Figure 2')
    model = mapper['model']
    ds, beta, alpha, gamma, timesteps = get_validation_set(mapper)

    #plt.plot(beta, label='beta')
    #plt.plot(alpha, label='alpha')
    ##plt.plot(sigma, label='sigma')
    #plt.plot(gamma, label='sigma')
    #plt.plot(np.sqrt(1-alpha), label='sqrt(1-alpha)')
    #plt.legend()
    #plt.xlabel('Time Step')
    #save_location = f'./plot_results/beta-dist.png'
    #plt.savefig(save_location)
    # might need to clean after

    #------

    img_num = 0
    for I, L in ds.take(2): # batch
        img_num += 1
        print(I.numpy().shape, L.numpy().shape)
        I = I[..., :3]
        L = tf.one_hot(L, depth=7)

        # Inference with loaded I/L
        TS = list(range(timesteps))
        stepdata = list(zip(TS, beta, alpha, gamma))
        stepdata.reverse()
        print(TS)
        # Random noise
        Z = np.random.normal(loc=0, scale=1.0, size=I.shape)
        print("SHAPE:", Z.shape)
        Zs = []

        # Loop over timesteps
        for ts, b, a, g in stepdata:
            for st in range(1): # time step
                Zs.append(Z)
                        
                # All examples get the same time index
                t_tensor = ts * np.ones(shape=(I.shape[0], 1))
                                            
                # Predict the noise
                delta = model.predict(x={'image_input': Z, 'time_input': t_tensor, 'label_input': L})

                print(np.min(delta[0,:,:,:]), np.max(delta[0,:,:,:]), np.mean(delta[0,:,:,:]), np.std(delta[0,:,:,:]))

                print(ts,a,b)
                
                # Adjust the image
                #Z = (Z - np.sqrt(1 - a) * delta) / np.sqrt(a)
                #Z = np.clip(Z, -1.0, 1.0)
                Z = Z/np.sqrt(1-b) - delta * b / (np.sqrt(1-a) * np.sqrt(1-b))
                #Z = np.maximum(Z, -1)
                #Z = np.minimum(Z, 1)
                
                
                if ts > 0:
                    #Z = np.clip(Z, -1.0, 1.0)
                    # Add exploratory noise
                    noise = np.random.normal(loc=0, scale=1.0, size=I.shape)
                    Z = Z + g * noise

        # Final step
        Zs.append(Z)


        # Visualization
        #   You will need to play with this

        i=0
        cols = 10
        fig, axs = plt.subplots(len(Zs)//cols+2, cols, figsize=(20,25))

        cl = np.argmax(L[i,:,:,:], axis=-1)
        axs[0,0].imshow(cl, vmax=6, vmin=0)
        axs[0,1].imshow(I[i,:,:,:])

        for j in range(cols):
            axs[0,j].set_xticks([])
            axs[0,j].set_yticks([])
            axs[-1,j].set_xticks([])
            axs[-1,j].set_yticks([])


        for j, Z in enumerate(Zs):
            axs[j//cols+1, j%cols].imshow(convert_image(Z[i,:,:,:]))
            axs[j//cols+1, j%cols].set_title(f"Step {j}", pad=10)
            axs[j//cols+1, j%cols].set_xticks([])
            axs[j//cols+1, j%cols].set_yticks([])

        #axs.axis('off')  # Hide axis ticks

        axs[0, 0].set_ylabel("Label Image", fontsize=12, labelpad=60, rotation=0, va='center')
        axs[-1, 0].set_ylabel("Generated Image", fontsize=12, labelpad=60, rotation=0, va='center')
        fig.suptitle('Diffusion Steps in Image Generation', fontsize=32)
        save_location = f'./plot_results/fig2_img_{img_num}.png'
        fig.savefig(save_location)



def generate_diffusion_image_results(mapper:dict[str, list[object]]):
    print('Generate Figure 3')
    model = mapper['model']
    ds, beta, alpha, gamma, timesteps = get_validation_set(mapper)

    final_images = []
    img_num = 0
    for I, L in ds.take(10):
        img_num += 1
        print(I.numpy().shape, L.numpy().shape)
        I = I[..., :3]
        L = tf.one_hot(L, depth=7)


        # Inference with loaded I/L
        TS = list(range(timesteps))
        stepdata = list(zip(TS, beta, alpha, gamma))
        stepdata.reverse()
        print(TS)
        # Random noise
        Z = np.random.normal(loc=0, scale=1.0, size=I.shape)
        print("SHAPE:", Z.shape)
        Zs = []

        # Loop over timesteps
        for ts, b, a, g in stepdata:
            for st in range(1):
                Zs.append(Z)
                        
                # All examples get the same time index
                t_tensor = ts * np.ones(shape=(I.shape[0], 1))
                                            
                # Predict the noise
                delta = model.predict(x={'image_input': Z, 'time_input': t_tensor, 'label_input': L})

                print(np.min(delta[0,:,:,:]), np.max(delta[0,:,:,:]), np.mean(delta[0,:,:,:]), np.std(delta[0,:,:,:]))

                print(ts,a,b)
                
                # Adjust the image
                #Z = (Z - np.sqrt(1 - a) * delta) / np.sqrt(a)
                #Z = np.clip(Z, -1.0, 1.0)
                Z = Z/np.sqrt(1-b) - delta * b / (np.sqrt(1-a) * np.sqrt(1-b))
                #Z = np.maximum(Z, -1)
                #Z = np.minimum(Z, 1)
                
                
                if ts > 0:
                    # Add exploratory noise
                    noise = np.random.normal(loc=0, scale=1.0, size=I.shape)
                    Z = Z + g * noise

        # Final step
        Zs.append(Z)
        final_images.append(Zs[-1])

    # Visualization
    #   You will need to play with this

    # Set up the figure
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    #cl = np.argmax(L[0,:,:,:], axis=-1)
    axs = axs.flatten()

    for i, img in enumerate(final_images):
        ax = axs[i]
        #plt.imshow(convert_image(Zs[-1][i,:,:,:]))
        ax.imshow(img[0,:,:,:])
        #final_image = (img + 1.0) / 2.0
        #final_image = tf.clip_by_value(final_image, 0.0, 1.0)
        #ax.imshow(final_image[0,:,:,:])
        #ax.imshow(img[0,:,:,:])
        ax.axis('off')  # Hide axis ticks

    fig.suptitle('Diffusion Image Generation Gallery')
    save_location = './plot_results/fig3.png'
    fig.savefig(save_location)



#################################################################
if __name__ == "__main__":
    mapper = get_file_content() # NOTE Needed to load models & model info

    generate_image_noise_over_time(mapper) # Figure 2
    generate_diffusion_image_results(mapper) # Figure 3
    print('done')
