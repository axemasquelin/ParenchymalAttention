# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Dependencies
# ---------------------------------------------------------------------------- #
from torch.utils.data import Dataset
from sklearn.utils import resample
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import glob
import nrrd
import math
import os
import cv2

import ParenchymalAttention.utils.image as image 
# ---------------------------------------------------------------------------- #

def seg_method(im, maskmap= None, method='Segmented', masksize=None):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    
    if method == 'OtsuMask':
        thres_img = otsu_algo(im, masksize)
    
    if method == 'BlockMask':
        thres_img = block_algo(im, masksize)

    if method =='Tumor-Segmentation':
        thres_img = segmentation_map(im, maskmap)

    if method =='Surround-Segmentation':
        maskmap = 1-maskmap
        thres_img = segmentation_map(im, maskmap)

    return thres_img

def otsu_algo(im, mask_size):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """

    thres_img = np.zeros((1,im.shape[0],im.shape[1]))
    mask = np.zeros((1,im.shape[0],im.shape[1]))

    _,thresh = cv2.threshold(im, 127, 255, cv2.THRESH_OTSU)         # Generating Otsu Threshold Map
    thres_img[0][:][:] = im                                         # Defining image that we will modify
    mask[0][:][:] = thresh                                          # Setting mask equal to Otsu Threshold output
    
    SP_x = int((im.shape[0] - mask_size)/ 2)                        # Defining Start X pixel, this is where the mask will start
    SP_y = int((im.shape[1] - mask_size)/2)                         # Defining Start Y pixel, this is wehere the mask will start

    for pixel_x in range(mask_size):                                # Defining x pixel location that will run through mask size
        for pixel_y in range(mask_size):                            # Defining y pixel location that will run through mask size

            if mask[0][pixel_x + SP_x][pixel_y + SP_y] != 0:        # If Pixel Value at start of mask location is not 0, 
                thres_img[0][pixel_x + SP_x][pixel_y + SP_y] = 0    # Set the value of the new threshold image to 0

            pixel_y += 1
        pixel_x += 1
    
    return thres_img

def block_algo(im, mask_size):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    thres_img = np.zeros((1,im.shape[0],im.shape[1]))
    thres_img[0][:][:] = im                                 # Defining image that we will modify
    
    SP_x = int((im.shape[0] - mask_size)/ 2)                # Defining Start X pixel, this is where the mask will start
    SP_y = int((im.shape[1] - mask_size)/2)                 # Defining Start Y pixel, this is wehere the mask will start

    for pixel_x in range(mask_size):                        # Defining x pixel location that will run through mask size
        for pixel_y in range(mask_size):                    # Defining y pixel location that will run through mask size

            thres_img[0][pixel_x + SP_x][pixel_y + SP_y] = 0    # Set the value of the new threshold image to 0

            pixel_y += 1
        pixel_x += 1
    
    return thres_img

def segmentation_map(im, maskmap):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    
    thres_img = np.zeros(im.shape)
    # print(im.shape)
    # print(maskmap.shape)
    for i in range(im.shape[0]):
        thres_img[i,:,:] = im[i,:,:] * maskmap[0]

    return thres_img

def normalize_img(img, normalization: str):
    '''
    Description:
    ----------
    Parameters:
    img - np.array
        raw image from nrrd file
    -------
    Outputs:
    img - np.array
        array containing a slice of the imag
    '''
    # print(img[0,:,:])
    img += abs(img.min())
    if normalization == 'norm':
        img = (img - img.min())/(img.max()-img.min())

    if normalization == 'stand':
        pixelmean = img.mean()
        pixelstd = img.std()
        img = (img - pixelmean)/(pixelstd)
        img = (img - img.min())/(img.max()-img.min())

    if normalization == 'lognorm':
        img = (np.log10(img) - np.log10(img).min())/(np.log10(img).max()-np.log10(img).min())

    if normalization == 'logstand':
        pixelmean = np.log10(img).mean()
        pixelstd = np.log10(img).std()
        img = (np.log10(img)-pixelmean)/pixelstd

    return img

def get_dims(df:pd.DataFrame, augment:str):
    """
    Get slice view of Nrrd file
    -----------
    Parameters:
    df - pd.Dataframe
        Panda dataframe containing paths for original nrrd and segmented nrrd
    augment - str
        string describing whether augmentation is random or inference
            random: randomly selects a slice based on a location
            central: repeats pattern to ensure selection of nearby centroid slices
    --------
    Returns:
    """
    slice_idx = np.zeros(len(df))
    bound1_lower = np.zeros(len(df))
    bound1_upper = np.zeros(len(df))
    bound2_lower = np.zeros(len(df))
    bound2_upper = np.zeros(len(df))

    for index in df.index:
        row = df.iloc[index]
        dim = [row['xdim'], row['ydim'], row['zdim']]
        roi = create_roi(row, dim, window_size=64)
        
        if row['view'] == 'x':
            slice_idx[index] = np.random.choice(np.arange(roi['Xmin'],roi['Xmax']))
            bound1_lower[index] = roi['Ymin']
            bound1_upper[index] = roi['Ymax']
            bound2_lower[index] = roi['Zmin']
            bound2_upper[index] = roi['Zmax']
        if row['view'] == 'y':
            slice_idx[index] = np.random.choice(np.arange(roi['Ymin'],roi['Ymax']))
            bound1_lower[index] = roi['Xmin']
            bound1_upper[index] = roi['Xmax']
            bound2_lower[index] = roi['Zmin']
            bound2_upper[index] = roi['Zmax']
        if row['view'] == 'z':
            slice_idx[index] = np.random.choice(np.arange(roi['Zmin'], roi['Zmax']))
            bound1_lower[index] = roi['Xmin']
            bound1_upper[index] = roi['Xmax']
            bound2_lower[index] = roi['Ymin']
            bound2_upper[index] = roi['Ymax']
    
    df['slice_idx'] = slice_idx
    df['bound1_lower'] = bound1_lower
    df['bound1_upper'] = bound1_upper
    df['bound2_lower'] = bound2_lower
    df['bound2_upper'] = bound2_upper

    return df
   
def scan_3darray(arr, view:str, threshold:int= 1):
    """
    Scan Array in a given view to get indices where nodule exists from mask
    -----------
    Parameters:
    arr
    --------
    Returns:
    x,y,z - list of position where nodule exists
    """ 
    edges=[]
    if view=='x':
        for i in range(arr.shape[0]):
            if arr[i,:,:].any() == threshold:
                edges.append(i)
    if view=='y':
        for i in range(arr.shape[1]):

            if arr[:,i,:].any() == threshold:
                edges.append(i)
    if view=='z':
        for i in range(arr.shape[2]):
            if arr[:,:,i].any() == threshold:
                edges.append(i)
    return edges