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
        thres_img = image.otsu_algo(im, masksize)
    
    if method == 'BlockMask':
        thres_img = image.block_algo(im, masksize)

    if method =='Tumor-Segmentation':
        thres_img = image.segmentation_map(im, maskmap)

    if method =='Surround-Segmentation':
        maskmap = 1-maskmap
        thres_img = image.segmentation_map(im, maskmap)

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
        array containing a slice of the image
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
    sliceview = [x.split('_')[2] for x in fileids]

    df = pd.DataFrame(np.transpose([originals, segmented, pid, ca, sliceview]), columns=['uri','segmented_uri', 'pid', 'ca', 'sliceview'])
    df['ca'] = df['ca'].astype(int)
    return df

def load_files(config):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """

    # Loading Original Data
    filelist = glob.glob(config['experiment']['data'] + 'Original/*.png')  
    pid = [os.path.basename(filename).split('_')[0] for filename in filelist]
    ca = [os.path.basename(filename).split('_')[1] for filename in filelist]
    sliceview = [os.path.basename(filename).split('_')[2].split('.')[0] for filename in filelist]
    dims = [np.asarray(Image.open(x)).shape for x in filelist]
    df = pd.DataFrame(np.transpose([pid,ca,sliceview, dims, filelist]), columns=['pid','ca','sliceview', 'dimension','uri'])

    # Loading Segmented Filenames
    filelist = glob.glob(config['experiment']['data'] + 'Segmented/*.png')
    pid = [os.path.basename(filename).split('_')[0] for filename in filelist]
    sliceview = [os.path.basename(filename).split('_')[2].split('.')[0] for filename in filelist]
    dims = [np.asarray(Image.open(x)).shape for x in filelist]
    df2 = pd.DataFrame(np.transpose([pid,sliceview,dims, filelist]), columns=['pid','sliceview','seg_dimension','segmented_uri'])

    df = df.merge(df2, how='right', on=['pid','sliceview'])
    df['ca'] = df['ca'].astype(int)
    df = df[df.dimension == (64,64,4)]
    df = df[df.seg_dimension == (64,64,4)]
    df = resample_df(config['experiment']['seed'], df, 2)

    return df

class DFLoader(Dataset):
    """
    """
    def __init__(self, data, method= None, augmentations=None, masksize=None, norms='stand'):
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

        maskmap = Image.open(row['segmented_uri'])
        maskmap = np.asarray(maskmap).T

        label = row['ca']
        im = np.zeros((1,64,64))
        if self.method != 'Original':
            im[0,:,:] = img[0,:,:]
            img = normalize_img(im, self.norms)
            img = seg_method(img, maskmap=maskmap, method= self.method, masksize = self.masksize)
            sample = {'image': img,
                    'label': label,
                    'id': row['pid']}

        else:
            im[0,:,:] = img[0,:,:]
            sample = {'image': normalize_img(im, self.norms),
                      'label': label,
                      'id': row['pid']}

        return sample         

