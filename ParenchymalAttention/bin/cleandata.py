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
        sys.stdout.write(f"\r {index+1} of {max_index} ~~~~~~ Progress: {((index+1)/max_index)*100:=5.3f} % ")

def save_nparray(img, segmentation, slice_idx:dict,
                pid:int, ca:int, savepath:str, ext:str='.npy'):
    """
    Saves the Region of Interest as a numpy 3 dimensional array.
    -----------
    Parameters:
    img - np.array()
    segmentation - np.array()
    slice_idx - dict
    """
    img = img[slice_idx['Xmin']:slice_idx['Xmax'],slice_idx['Ymin']:slice_idx['Ymax'], slice_idx['Zmin']:slice_idx['Zmax']]
    segmentation = segmentation[slice_idx['Xmin']:slice_idx['Xmax'],slice_idx['Ymin']:slice_idx['Ymax'], slice_idx['Zmin']:slice_idx['Zmax']]
    
    if img.shape[0] != img.shape[1] or img.shape[0] != img.shape[2] or img.shape[1] != img.shape[2]:
        print(f'\nPID: {pid}, Image Shape: {img.shape}, Segmentation shape: {segmentation.shape}')
    
    np.save(savepath + '/Original/' + str(pid) + '_' + str(ca) + ext, img)
    np.save(savepath + '/Segmented/' + str(pid) + '_' + str(ca) + ext, segmentation)

def validate(df, config):
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
            if all(img[1]['sizes'] == seg_map[1]['sizes']):
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
                slice_idx = pre.create_roi(img[1]['sizes'], x, y, z, window_size=64)
                save_nparray(img=img[0], segmentation=seg_map[0], slice_idx=slice_idx,
                                pid=row['pid'], ca=row['ca'], savepath=config['savedirectory'])

            else:
                '''Error Code for Segmentation Map not equal to Original Image'''
                valid[idx] = 2
            
        except KeyboardInterrupt:
            break
        except:
            pass
            # print(f'\nFailed to load pid: {idx}\n')

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

    print(f'   Category     |  Dataset Percentage      ')        
    print(f'Percent Valid   |{(len(df[df.valid==1])/len(df))*100:>5.2f}' \
          f'\n Size Issues    | {(len(df[df.valid==2])/len(df))*100:>5.2f}' \
          f'\nLoading Issues: | {(len(df[df.valid==0])/len(df))*100:>5.2f}')
    
    df = df[df.valid == 1]

    print(f'Number of Benign {len(df[df.ca==0])} | Number of Malignant: {len(df[df.ca==1])}')
    
    return df

def load_files(config:dict, augment:int=1):
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
    df = validate(df, config)
    
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
    df = load_files(config, augment= 3)
    # df.to_csv(config['savedirectory'] + '/npylist.csv')
    

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
    