# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Dependencies
# ---------------------------------------------------------------------------- #
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from PIL import Image

import numpy as np
import image 

import glob, os, cv2
# ---------------------------------------------------------------------------- #

def nodule_remove(im, mask_size, normalize = False, masktype = 'OtsuMask'):
    """
    Definition:
    Input:
        -im : Image 
        -normalize: normalize pixel values flag (True or False)
        -ms: mask size for otsu algorithm. 
    Output: 
        -thres_img: thresholded image, returns the masked image
    """
    
    if masktype == 'OtsuMask':
        thres_img = image.otsu_algo(im, mask_size)
    
    if masktype == 'BlockMask':
        thres_img = image.block_algo(im, mask_size)

    if normalize:
        thres_img = normalize_img(thres_img)

    return thres_img


def normalize_img(X):
    '''
        Description: Import dataset into a pandas dataframe and filter out empty data entires
        
    '''
    for i in range(len(X)):
        img = X[i]
        maxHU = np.max(img)                  # Find maximum pixel value of image
        minHU = np.min(img)                  # Find minimum pixel value of image
    
        norm = (img - minHU) / (maxHU - minHU)      # calculate normalized pixel values
        norm[norm>1] = 1                            # If Normal is greater than 1, set to 1
        norm[norm<0] = 0                            # if norm is less than zero set value to 0

        X[i] = img
    
    return X


def resample_img(
        neg_img, pos_img,
        neg_label, pos_label,
        neg_name, pos_name,
        method
    ):
    '''
        Equalize the number of samples in each class:
        -- method = 1 - upsample the positive cases
        -- method = 2 - downsample the negative cases
    '''
    len_neg = len(neg_img)    # Benign Nodule
    len_pos = len(pos_img)    # Malignant Nodule

    # Upsample the pos samples
    if method == 1:
        pos_upsampled, pos_label, pos_name = resample(
            pos_img, pos_label, pos_name,
            n_samples=len_neg,
            replace=True, 
            random_state=10
        )

        return np.concatenate([pos_upsampled, neg_img]), np.concatenate([pos_label, neg_label]), np.concatenate([pos_name, neg_name])

    # Downsample the neg samples
    elif method == 2:
        neg_downsampled, neg_label, neg_name = resample(
            neg_img, neg_label, neg_name,
            n_samples=len_pos,
            replace=True, 
            random_state=10
        )

        return np.concatenate([pos_img, neg_downsampled]), np.concatenate([pos_label, neg_label]), np.concatenate([pos_name, neg_name])

    else:
        print('Error: unknown method')
        
def load_img(
    folder,                 # Str for dataset folder locations
    masktype = None,        # Temporary Variable, currently not used will change structure in future
    normalize = True,       # Variable to control whether the data is normalized or not
    resample = 2,           # Resample value (1 -> up sample, 2 -> down-sample)
    mask_size = 64,         # Define mask size applied to image, This may get modified
    seed = 2022,
    ):
    '''
        Returns:
    '''
    filenames = glob.glob(folder + '*.jpg')

    neg_img = []
    pos_img = []

    neg_label = []
    pos_label = []

    neg_name = []
    pos_name = []

    for filename in filenames:
        
        im = np.zeros((1,64,64))
        ca = os.path.splitext(filename.split('_')[-1])[0]
        
        img = cv2.imread(filename, 0)
        
        if masktype == 'Original':
            im[0,:,:] = np.array(img)
        
            if normalize:
                im = normalize_img(im)

        else:
            im = nodule_remove(img, mask_size, normalize= normalize, masktype= masktype)

        if ca == '1':
            pos_name.append(filename)
            pos_img.append(im)
            pos_label.append(1) 
        
        else: 
            neg_name.append(filename)
            neg_img.append(im)
            neg_label.append(0)
    
    X, y, x_names = resample_img(neg_img, pos_img,
                        neg_label, pos_label,
                        neg_name, pos_name,
                        resample)
    
    return X, y, x_names
