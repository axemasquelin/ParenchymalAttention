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
    # print(im.shape)
    thres_img = np.zeros(im.shape)
    # print(thres_img.shape)
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

def create_bounds(arr, window_size = 64):
    pass


def scan_3darray(arr, threshold= 1):
    """
    Description:
    -----------
    Parameters:
    arr - np.array()
    threshold - int
        Value to identify edges of masks
    --------
    Returns:
    x,y,z - list of position where nodule exists
    """ 
    x,y,z= [],[],[]

    for i in range(arr.shape[0]):
        if arr[i,:,:].any() >= threshold:
            x.append(i)

    for i in range(arr.shape[1]):
        if arr[:,i,:].any() >= 1:
            y.append(i)
    
    for i in range(arr.shape[2]):
        if arr[:,:,i].any() >= 1:
            z.append(i)
    
    return x,y,z

def get_slices(data, slice_idx, pid, ca, segmented, savepath):
    """
    Description:
    -----------
    Parameters:
    args - dict
    --------
    Returns:
    """
    xslice = data[slice_idx['Xmid'], slice_idx['Ymin']:slice_idx['Ymax'], slice_idx['Zmin']:slice_idx['Zmax']]
    yslice = data[slice_idx['Xmin']:slice_idx['Xmax'], slice_idx['Ymid'], slice_idx['Zmin']:slice_idx['Zmax']]
    zslice = data[slice_idx['Xmin']:slice_idx['Xmax'], slice_idx['Ymin']:slice_idx['Ymax'], slice_idx['Zmid']] 