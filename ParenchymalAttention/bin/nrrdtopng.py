# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #
from argparse import Namespace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nrrd
import math
import argparse
import logging
import cv2
import sys, os, glob
# ---------------------------------------------------------------------------- #

def load_files(config):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """ 
    masked_filelist = glob.glob(config['masksloc']+'*.nrrd')
    pid = [os.path.basename(x).split('_')[0] for x in masked_filelist]
    df1 = pd.DataFrame(np.transpose([pid,masked_filelist]), columns=['pid','segmented_file'])
    
    benign_filelist = glob.glob(config['benignloc']+'*.nrrd')
    pid = [os.path.basename(x).split('_')[0] for x in benign_filelist]
    ca = [0 for x in benign_filelist]
    begn_df = pd.DataFrame(np.transpose([pid,benign_filelist,ca]), columns=['pid','Original_file','ca'])
    
    malignant_filelist = glob.glob(config['malignantloc']+'*.nrrd')
    pid = [os.path.basename(x).split('_')[0] for x in malignant_filelist]
    ca = [1 for x in malignant_filelist]
    mal_df = pd.DataFrame(np.transpose([pid,malignant_filelist,ca]), columns=['pid','Original_file','ca'])
    
    original_df = pd.concat([begn_df, mal_df])


    combined_df = pd.merge(df1, original_df, on='pid')

    return combined_df
    
   
def scan_3darray(arr, threshold= 1):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    x,y,z - list of position where nodule exists
    """ 
    x,y,z= [],[],[]

    for i in range(arr.shape[0]):
        if arr[i,:,:].any() >= threshold:
            x.append(i)

    for i in range(arr.shape[1]):
        if arr[:,i,:].any() >= threshold:
            y.append(i)
    for i in range(arr.shape[2]):
        if arr[:,:,i].any() >= threshold:
            z.append(i)
    return x,y,z

def check_savedir(savepath, segmented):
    """
    Description:
    -----------
    Parameters:
    args - dict
    --------
    Returns:
    """
    if segmented:
        savepath = savepath + '/Segmented/'
    else:
        savepath = savepath + '/Original/'

    dir_exists = os.path.exists(savepath)
    if not dir_exists:
        sys.stdout.write('\n\r {0} | Creating Result Directories | {0}\n '.format('-'*50))
        print(savepath)
        os.mkdir(savepath)
    
    return savepath

def save_slice(data:np.array,savepath:str, pid:int, ca:int, slice:str, ext:str= '.png'):
    """
    Saves Slice of the NRRD Data
    -----------
    Parameters:
    data - np.array()
        Contains the pixel/voxel values for a given slice from the nrrd image
    savepath - str
        directory in which the image will be stored
    pid - int
        patient identification value to be traced back to original nrrd file
    ca - int
        classification of the nodule (1 = malignant, 0 = benign)
    slice - str
        slice view of the nodule (x,y,z)
    ext - str
    """
    filename = savepath + '/' + str(pid) + '_' + str(ca) + '_' + slice + ext
    if data.shape == [64,64]:
        cv2.imwrite(filename, data)
    
def get_slices(data, slice_idx, pid, ca, segmented, savepath):
    """
    Description:
    -----------
    Parameters:
    args - dict
    --------
    Returns:
    """
    savepath = check_savedir(savepath, segmented)
    # print(slice_idx)
    xslice = data[slice_idx['Xmid'], slice_idx['Ymin']:slice_idx['Ymax'], slice_idx['Zmin']:slice_idx['Zmax']]
    yslice = data[slice_idx['Xmin']:slice_idx['Xmax'], slice_idx['Ymid'], slice_idx['Zmin']:slice_idx['Zmax']]
    zslice = data[slice_idx['Xmin']:slice_idx['Xmax'], slice_idx['Ymin']:slice_idx['Ymax'], slice_idx['Zmid']] 
    

    save_slice(xslice, savepath, pid, ca, slice = 'x', ext = '.png')
    save_slice(yslice, savepath, pid, ca, slice = 'y', ext = '.png')
    save_slice(zslice, savepath, pid, ca, slice = 'z', ext = '.png')
    
def create_bounds(shape:int, window_size:int, lower_val:int, upper_val:int):
    """
    -----------
    Parameters:
    shape - int
        Shape of the input image to verify where the edge of the image exists
    window_size - int
        prefered window size for selected object
    lower_val - int
        lowest idx value of an object in a given dimension
    upper_val - int
        highest idx value of an object in a given dimension
    --------
    Returns:
    pad_lower - int
        Modified lower bound to include necessary padding for image slice to fit window size
    pad_upper - int
        Modified upper bound to include necessary padding for image slice to fit window size
    mid_point - int
        Middle point of the object calculated based on upper and lower edge of object
    """

    pad = (window_size - (upper_val-lower_val))
    pad_lower = lower_val - (pad/2)
    pad_upper = upper_val + (pad/2)
    mid_point = lower_val + (upper_val-lower_val) / 2

    if (pad_upper >= shape):
        pad_lower -= (pad_upper - shape)
        if pad_lower < 0:
            pad_lower = 0
        pad_upper = shape
    if (pad_lower <= 0):
        pad_upper += (0 - pad_lower)
        if pad_upper > shape:
            pad_upper = shape
        pad_lower = 0

    return math.floor(pad_lower), math.floor(pad_upper), math.floor(mid_point)  

def create_roi(img_shape:list, x:list,y:list,z:list, window_size:int=64):
    """
    Create the Region of Interest in which the nodule exists and identifies the middle
    points \n 
    ----------- \n
    Parameters: \n
    row - pd.DataSeries
        Contains the information for the Nrrd files
    window_size - int
        Size of the Region of interest
    --------
    Returns:
    """  

    xlower, xupper, xmid = create_bounds(img_shape[0], window_size, x[0], x[-1])
    ylower, yupper, ymid = create_bounds(img_shape[1], window_size, y[0], y[-1])
    zlower, zupper, zmid = create_bounds(img_shape[2], window_size, z[0], z[-1])
    
    slice_idx = {
        'Xmin': xlower,
        'Xmid': xmid,
        'Xmax': xupper,
        'Ymin': ylower,
        'Ymid': ymid,
        'Ymax': yupper,
        'Zmin': zlower,
        'Zmid': zmid,
        'Zmax': zupper,
    }
    
    return slice_idx

def main(args, command_line_args):
    """
    Description:
    -----------
    Parameters:
    args - dict
    --------
    Returns:
    """ 
    config = {
        'savedirectory': args.savepath,     # Savepath for final Directory
        'masksloc': args.maskpath,          # Segmented Nodule's Nrrd Location
        'benignloc': args.benignpath,       # Benign Nodule Location
        'malignantloc': args.malignantpath  # Malignant Nodule Location
    }

    df = load_files(config)

    for idx in df.index:
        row = df.iloc[idx]
        try:
            seg_data = nrrd.read(row['segmented_file'])
            ori_data = nrrd.read(row['Original_file'])
            x,y,z = scan_3darray(arr=seg_data[0], threshold = 1)

            slice_idx = create_roi(seg_data[0].shape, x,y,z, window_size=64)
            
            get_slices(data=ori_data[0], slice_idx=slice_idx, pid=row['pid'], ca=row['ca'], segmented = False, savepath = config['savedirectory'])
            get_slices(data=seg_data[0], slice_idx=slice_idx, pid=row['pid'], ca=row['ca'], segmented = True, savepath = config['savedirectory'])

        except:
            print("WARNING: Error Loading files for Pid - %s"%row['pid'])
            break
    
def build_parser() -> argparse.ArgumentParser:
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """ 
    parser = argparse.ArgumentParser()

    parser.add_argument('--savepath', type=str, required = True, help= 'save location for png')
    parser.add_argument('--maskpath', type=str, required = True, help= 'directory of Masked Nodules Nrrd Files')
    parser.add_argument('--benignpath', type=str, required = True, help= 'directory of Benign Nodules Nrrd Files')
    parser.add_argument('--malignantpath', type=str, required = True, help= 'directory of Malignant Nodules Nrrd Files')
    
    return parser

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()
    main(args= args, command_line_args= sys.argv)
    