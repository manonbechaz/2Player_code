from tqdm import tqdm
from skimage.segmentation import slic
from skimage.measure import regionprops
import pandas as pd
import torch.nn as nn
import numpy as np 
import math
from numpy.linalg import norm
from data_processing.dataset_HRSCD import *
from data_processing.dataset_ValaisCD import *
from data_processing.dataset_LEVIRCD import *
from data_processing.dataset_WHU import *
from argparse import ArgumentParser

def compute_similarity(pixel1, pixel2, sigma = 100):
    norm = np.linalg.norm(pixel1-pixel2)
    return math.exp(-sigma*(norm**2))
    

def compute_SDSN(df,list_pixels):
    vals = []
    for i, row in df.iterrows():
        val_row = row['mean_vals']
        f = np.zeros(len(list_pixels))
        j = 0
        for pixel in list_pixels:
            pix_val = np.array(pixel,dtype='float32')
            f[j] = compute_similarity(val_row,pix_val)
            j = j+1

        vals.append(f)
    
    df['SDSN']=vals

def find_closest_points(df1, df2):
    closest_values = []  # Store closest 'value' from df2 for each point in df1
    # Iterate over each point in df1
    for i, row1 in df1.iterrows():
        # Calculate Euclidean distance between the current point in df1 and all points in df2
        distances = np.sqrt((df2['x']- row1['x'])**2 + (df2['y'] - row1['y'])**2)
        
        # Get the index of the minimum distance
        min_index = distances.idxmin()
        
        # Append the closest 'value' from df2 to the list
        closest_values.append(df2.loc[min_index, 'SDSN'])

    # Add the closest values as new columns in df1
    df1['closest_SDSN_in_other_im'] = closest_values  # Closest value array from df2


def compute_probability_map(i1, i2, lab, nsuperpix = 200, kernel_size_avg=64):
    # Image 1, 2 in torch
    i1_numpy = i1.permute(1, 2, 0).numpy()
    i2_numpy = i2.permute(1, 2, 0).numpy()

    avg_pool = nn.AvgPool2d(kernel_size=kernel_size_avg, stride=kernel_size_avg)
    segmented_im1 = slic(i1_numpy,n_segments=nsuperpix)
    segmented_im2 = slic(i2_numpy,n_segments=nsuperpix)

    downsampled_im1 = avg_pool(i1).permute(1, 2, 0).numpy()
    downsampled_im2 = avg_pool(i2).permute(1, 2, 0).numpy()

    regions1 = regionprops(segmented_im1, intensity_image=i1.permute(1, 2, 0).numpy())
    regions2 = regionprops(segmented_im2, intensity_image=i2.permute(1, 2, 0).numpy())

    centroids1 = [props.centroid for props in regions1]
    centroids2 = [props.centroid for props in regions2]
    mean_vals1 = [props.mean_intensity for props in regions1]
    mean_vals2 = [props.mean_intensity for props in regions2]

    df1 = pd.DataFrame({'centroids':centroids1, 'mean_vals': mean_vals1})
    df1['x']=df1['centroids'].apply(lambda x: x[0])
    df1['y']=df1['centroids'].apply(lambda x: x[1])

    df2 = pd.DataFrame({'centroids':centroids2, 'mean_vals': mean_vals2})
    df2['x']=df2['centroids'].apply(lambda x: x[0])
    df2['y']=df2['centroids'].apply(lambda x: x[1])

    # downscaled image
    pixel_list_im1 = downsampled_im1.reshape(int((256/kernel_size_avg)**2), 3)
    pixel_list_im2 = downsampled_im2.reshape(int((256/kernel_size_avg)**2), 3)

    # Convert it to a list if needed (it is an array by default)
    pixel_list_im1 = pixel_list_im1.tolist()
    pixel_list_im2 = pixel_list_im2.tolist()

    compute_SDSN(df1,pixel_list_im1)
    compute_SDSN(df2,pixel_list_im2)

    find_closest_points(df1, df2)
    find_closest_points(df2, df1)

    df1['dissimilarity']=df1.apply(lambda x: np.dot(x['SDSN'],x['closest_SDSN_in_other_im'])/(norm(x['SDSN'])*norm(x['closest_SDSN_in_other_im'])),axis=1)
    df2['dissimilarity']=df2.apply(lambda x: np.dot(x['SDSN'],x['closest_SDSN_in_other_im'])/(norm(x['SDSN'])*norm(x['closest_SDSN_in_other_im'])),axis=1)

    df1['dissimilarity'] = 1-(df1['dissimilarity']+1)/2
    df2['dissimilarity'] = 1-(df2['dissimilarity']+1)/2

    scores1 = list(df1['dissimilarity'])
    scores2 = list(df2['dissimilarity'])

    confidence_map1 = np.zeros_like(lab.squeeze().numpy(),dtype=float)
    for region, s in zip(regions1,scores1):
        for coords in region.coords:  # For each pixel in the region, color it
            confidence_map1[coords[0], coords[1]] = s

    confidence_map2 = np.zeros_like(lab.squeeze().numpy(),dtype=float)
    for region, s in zip(regions2,scores2):
        for coords in region.coords:  # For each pixel in the region, color it
            confidence_map2[coords[0], coords[1]] = s

    #confidence_map = np.maximum(confidence_map1,confidence_map2)
    confidence_map = (confidence_map1+confidence_map2)/2

    return confidence_map

def compute_save_all_prob_maps(dataset, type='train', folder_path=None):

    if dataset == 'HRSCD':
        ds_train = HRSCD_Dataset(type = type,cropsize=256,changed_only=False, refined_labels=True, name=True)
    elif dataset == 'Valais':
        ds_train = Valais_Dataset(type = type,cropsize=256, name=True, probability_maps = False)
    elif dataset == 'LEVIR':
        ds_train = LEVIRCD_Dataset(cropsize=1024, type=type, name=True)
    elif dataset == 'WHUCD':
        ds_train = WHUCD_Dataset(cropsize=256, type=type,probability_maps=False, name=True)
    
    for i in tqdm(range(len(ds_train))):
        i1,i2,lab,name = ds_train[i]
        probability_map = compute_probability_map(i1,i2,lab,nsuperpix=100)
        file = name.split('/')[-1]
        saving_name = folder_path + '/probability/'+file[:-4]+'.npy'
        with open(saving_name, 'wb') as f:
            np.save(f,probability_map)

if __name__ == "__main__": 

    parser = ArgumentParser()
    parser.add_argument("--dataset", type = str)
    args = parser.parse_args()
    
    if args.dataset == 'HRSCD':
        folder_path_train = PATH_TRAIN_DS_HRSCD
        folder_path_test = PATH_TEST_DS_HRSCD
        folder_path_val = PATH_VAL_DS_HRSCD

    if args.dataset == 'Valais':
        folder_path_train = PATH_TRAIN_DS_VALAIS
        folder_path_test = PATH_TEST_DS_VALAIS
        folder_path_val = PATH_VAL_DS_VALAIS

    if args.dataset == 'LEVIR':
        folder_path_train = PATH_TRAIN_LEVIR
        folder_path_test = PATH_TEST_LEVIR
        folder_path_val = PATH_VAL_LEVIR

    if args.dataset == 'WHUCD':
        folder_path_train = PATH_TRAIN_WHU
        folder_path_test = PATH_TEST_WHU
        folder_path_val = PATH_VAL_WHU

    print('Computing for the training set...')
    compute_save_all_prob_maps(args.dataset, type='train',folder_path=folder_path_train)
    
    print('Computing for the validation set...')
    compute_save_all_prob_maps(args.dataset, type='val',folder_path=folder_path_train)
    
    print('Computing for the test set...')
    compute_save_all_prob_maps(args.dataset, type='test',folder_path=folder_path_train)





        


    