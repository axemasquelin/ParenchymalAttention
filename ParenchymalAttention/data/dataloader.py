# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    ---------------------------------------
    Functions related to pytorch dataloader and generating dataframe for dataloader.
    Includes inference dataloader (InferLoader), Nrrd dataloader (NrrdLoader) and a
    csv dataloader (CsvLoader). Image information and location is expected to exist in a 
    csv file, and include a 'uri' column (original image filepath), segmented_uri (segmentation maps filepath)
    pid (patient identifier), and ca (classification).

'''
# Dependencies
# ---------------------------------------------------------------------------- #
from torch.utils.data import Dataset
from sklearn.utils import resample
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import random
import glob
import nrrd
import os
import cv2

import ParenchymalAttention.utils.image as image 
import ParenchymalAttention.data.preprocess as prep
# ---------------------------------------------------------------------------- #
   
def load_files(config):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    df = pd.read_csv(config['experiment']['data'])
    df = resample_df(config['experiment']['seed'], df, 2)
    
    return df

def resample_df(seed, df, method):
    '''
        Equalize the number of samples in each class:
        -- method = 1 - upsample the positive cases
        -- method = 2 - downsample the negative cases
        -- method = 3 - return all cases
    '''
    df_neg = df[df.ca==0]
    df_pos = df[df.ca==1]

    # Upsample the pos samples
    if method == 1:
        df_pos_upsampled = resample(
            df_pos,
            n_samples=len(df_neg),
            replace=True, 
            random_state= seed
        )
        return pd.concat([df_pos_upsampled, df_neg])

    # Downsample the neg samples
    elif method == 2:
        df_neg_downsampled = resample(
            df_neg,
            n_samples=len(df_pos),
            replace=True, 
            random_state= seed
        )
        return pd.concat([df_pos, df_neg_downsampled])
    
    # Return Raw Dataset
    elif method == 3:
        return pd.concat([df_pos, df_neg])

    else:
        print('Error: unknown method')

def cherrypick(fileids:list, filepath:str):
    """
    Description:
    -----------
    Parameters:
    -------
    Returns:
    """
    originals = [filepath + 'Original/' + x for x in fileids]
    segmented = [filepath + 'Segmented/' + x for x in fileids]
    pid = [x.split('_')[0] for x in fileids]
    ca = [x.split('_')[1] for x in fileids]
    dims = [np.asarray(Image.open(x)).shape for x in originals]
    sliceview = [x.split('_')[2].split('.')[0] for x in fileids]

    df = pd.DataFrame(np.transpose([originals, segmented, pid, ca, sliceview, dims]), columns=['uri','segmented_uri', 'pid', 'ca', 'sliceview', 'dimension'])
    df = df[df.dimension == (64,64,4)]
    print(df)
    df['ca'] = df['ca'].astype(int)

    return df



def inference_views(df:pd.DataFrame):

    for index in df.index:
        row = df.iloc[index]
        dim = [row['xdim'], row['ydim'], row['zdim']]
        x = [row['x_min'], row['x_max']]
        y = [row['y_min'], row['y_max']]
        z = [row['z_min'], row['z_max']]


    return df

def augment_dataframe(df, upsample:int=3,  augment:str='rand'):
    """
    Creates an augmented dataframe that randomly augments the dataset by taking slices adjacent to central slices, or takes all surrounding slices. 
    -----------
    Parameters:
    df - pandas.dataframe()
    upsample - int
        Number of slices to augment the dataset by
    augment - str
        type of augmentation employed by the function. Random will randomly select slices from the images
    --------
    Returns:
    df - pandas.dataframe()
    """

    if augment=='rand':
        df.loc[df.index.repeat(df.pid)].reset_index(drop=True)    
        df['view'] = [np.random.choice(['x','y','z'] for index in df.index)]
    else:
        sliceloc = inference_views(df)
        df.loc[df.index.repeat(df.pid)].reset_index(drop=True)  
    return df
class DFLoader(Dataset):
    """
    """
    def __init__(self, data, method= None, augmentations=None, masksize=None, norms='norm'):
        super().__init__()
        self.data = data
        self.augment= augmentations
        self.masksize = masksize
        self.norms = norms
        self.method = method

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index:int):
        row = self.data.iloc[index]
        img = Image.open(row['uri'])
        img = np.asarray(img).T     

        # maskmap = Image.open(row['segmented_uri'])
        maskmap = np.asarray(maskmap).T

        label = row['ca']
        im = np.zeros((1,64,64))
        if self.method != 'Original':
            im[0,:,:] = img[0,:,:]
            img = prep.normalize_img(im, self.norms)
            img = prep.seg_method(img, maskmap=maskmap, method= self.method, masksize = self.masksize)
            sample = {'image': img,
                    'label': label,
                    'id': row['pid']}

        else:
            im[0,:,:] = img[0,:,:]
            sample = {'image': prep.normalize_img(im, self.norms),
                      'label': label,
                      'id': row['pid']}

        return sample         

class NrrdLoader(Dataset):
    """
    Custom Nrrd file Dataloader class for pytorch; Will utilize raw and segmented Nrrd files to select random
    regions of the image for analysis.
    """
    def __init__(self, data, method:str=None, augmentation:str=None, masksize:int=None, norms:str='norm'):
        super().__init__()
        self.data = data
        self.augment = augmentation
        self.masksize = masksize
        self.norms = norms
        self.method = method

    def __len__(self)->int:
        return len(self.data)

    def __getitem__(self, index:int):
        row = self.data.iloc[index]
        img = nrrd.read(row['uri'])
        thres = nrrd.read(row['thresh_uri'])
        print(img)
        print(img[0].shape)
        print(thres[0].shape)

        if row['augment'] == 'rand':
            if row.view == 'x':
                slice_idx = np.random.choice(np.arange(row['x_min'],row['x_max']))

            elif row.view == 'y':
                slice_idx = np.random.choice(np.arange(row['y_min'],row['y_max']))

            else:
                slice_idx = np.random.choice(np.arange(row['z_min'],row['z_max']))

        elif row['augment'] == 'infer':
            pass
        else:

            sample = {'image': prep.normalize_img(img, self.norms),
                'label': row['ca'],
                'id': row['pid']}
            
        return sample

