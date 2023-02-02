# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Dependencies
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

import nrrdtopng as pre
# import ParenchymalAttention.utils.image as image 
# ---------------------------------------------------------------------------- #
def progress(index, max_index):
        sys.stdout.write(f"\r Current Index {index+1} | Max Index: {max_index}  ~~~~~~ Progress: {((index+1)/max_index)*100:=5} % ")

def validate(df):
    """
    Validation to ensure files can be properly opened and contains data (checking if corrupt)
    -----------
    Parameters:
    df - dataframe
        dataframe containing the file path, and classification
    --------
    Returns:
    """
    max_index = len(df)
    valid = np.zeros(max_index)
    x_min = np.zeros(max_index)
    x_max = np.zeros(max_index)
    y_min = np.zeros(max_index)
    y_max = np.zeros(max_index)
    z_min = np.zeros(max_index)
    z_max = np.zeros(max_index)
    xdim = np.zeros(max_index)
    ydim = np.zeros(max_index)
    zdim = np.zeros(max_index)
    for idx in df.index:

        row = df.iloc[idx]
        progress(idx,max_index)
        try:
            img = nrrd.read(row['uri'])
            seg_map = nrrd.read(row['thresh_uri'])

            x,y,z = pre.scan_3darray(arr=seg_map[0], threshold = 1)  
            xdim[idx] = img[0].shape[0]
            ydim[idx] = img[0].shape[1]
            zdim[idx] = img[0].shape[2]
            x_min[idx] = x[0]
            y_min[idx] = y[0]
            z_min[idx] = z[0]
            x_max[idx]= x[-1]
            y_max[idx]= y[-1]
            z_max[idx]= z[-1]
            valid[idx] = 1
        except:
            print(f'\nFailed to load pid: {idx}\n')
            valid[idx] = 0

    print(len(valid))
    df['valid'] = valid
    df['xdim'] = xdim
    df['x_min'] = x_min
    df['x_max'] = x_max
    df['ydim'] = ydim
    df['y_min'] = y_min
    df['y_max'] = y_max
    df['zdim'] = zdim
    df['z_min'] = z_min
    df['z_max'] = z_max
    df = df[df.valid == 1]

    
    return df

def load_files(config):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    # Loading Benign Nrrds
    filelist = glob.glob(config['benignloc']+ '*.nrrd')   
    pid = [os.path.basename(filename).split('_')[0] for filename in filelist]
    ca = [0 for filename in filelist]
    df_begn = pd.DataFrame(np.transpose([pid, ca, filelist]), columns=['pid','ca','uri'])

    # Loading Cancer Nrrds
    filelist = glob.glob(config['malignantloc']+ '*.nrrd')   
    pid = [os.path.basename(filename).split('_')[0] for filename in filelist]
    ca = [1 for filename in filelist]
    df_mal = pd.DataFrame(np.transpose([pid, ca, filelist]), columns=['pid','ca','uri'])
    df = pd.concat([df_begn, df_mal])

    # # Loading Segmented Filenames
    filelist = glob.glob(config['masksloc'] + '*.nrrd')
    pid = [os.path.basename(filename).split('_')[0] for filename in filelist]
    df2 = pd.DataFrame(np.transpose([pid, filelist]), columns=['pid','thresh_uri'])
    df = df.merge(df2, how='right', on=['pid'])

    # df = df.head(20)
    df = validate(df)
    
    return df

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
    window_size = 64
    df = load_files(config)
    df.to_csv(config['savedirectory'] + '/nrrdlist.csv')

def build_parser() -> argparse.ArgumentParser:
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """ 
    parser = argparse.ArgumentParser()

    parser.add_argument('--savepath', type=str, required = True, help= 'save location csv file')
    parser.add_argument('--maskpath', type=str, required = True, help= 'directory of Masked Nodules Nrrd Files')
    parser.add_argument('--benignpath', type=str, required = True, help= 'directory of Benign Nodules Nrrd Files')
    parser.add_argument('--malignantpath', type=str, required = True, help= 'directory of Malignant Nodules Nrrd Files')
    
    return parser

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()
    main(args= args, command_line_args= sys.argv)
    