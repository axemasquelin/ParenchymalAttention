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
from itertools import cycle
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
   
def load_files(config, ext:str='.csv'):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    if ext== '.csv':
        df = pd.read_csv(config['experiment']['data'])
        df = resample_df(config['experiment']['seed'], df, 2)
    
    if ext== '.npy': 
        original_filelist = glob.glob('./dataset/Original/'+'*.npy')
        pid = [os.path.basename(x).split('_')[0] for x in original_filelist]
        ca = [os.path.basename(x).split('_')[1].split('.')[0] for x in original_filelist]
        df = pd.DataFrame(np.transpose([pid,original_filelist,ca]), columns=['pid','uri', 'ca'])
        
        masked_filelist = glob.glob('./dataset/Segmented/'+'*.npy')
        pid = [os.path.basename(x).split('_')[0] for x in masked_filelist]
        df2 = pd.DataFrame(np.transpose([pid,masked_filelist]), columns=['pid','thresh_uri'])
        
        df = pd.merge(df, df2, on='pid')
        df['ca'] = df['ca'].astype(int)
        df = resample_df(seed=2022, df=df, method=2)

    else:
        print('WARNING: Filetype not compatible')

    return df

def check_columns(df:pd.DataFrame):
    """
    Checks for necessary column names to exist and unwanted column names
    -----------
    Parameters:
    df - pandas.DataFrame()
    --------
    Returns:
    df - pandas.DataFrame()
        Cleaned dataframe which excludes unamed columns    
    """
    expected_columns = ['uri','pid','ca','thres_uri','xdim','ydim','zdim']
    columns = df.columns.values.tolist()
    
    for column in columns:
        if column != expected_columns:
            df= df.drop(column, axis = 1)
    
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
    df['ca'] = df['ca'].astype(int)

    return df


def augment_dataframe(df:pd.DataFrame, upsample:int=3,  augment:str='rand'):
    """
    Creates an augmented dataframe that randomly augments the dataset by taking slices adjacent to central slices, or takes all surrounding slices. 
    \n
    -----------
    Parameters: \n
    df - pandas.dataframe()
        Pandas dataframe containing Nrrd list
    upsample - int
        Number of slices to augment the dataset by
    augment - str
        type of augmentation employed by the function. Random will randomly select slices from the images
    --------
    Returns: \n
    df - pandas.dataframe()
    """

    if augment=='rand':
        df = df.loc[df.index.repeat(upsample)].reset_index(drop=True)  
        # print(len(df))
        df['view'] = [np.random.choice(['x','y','z']) for index in df.index]
        # df['view'] = ['z' for index in df.index]
        # df = prep.get_dims(df, augment)

    else:
        views = cycle(['x','y','z'])
        df = df.loc[df.index.repeat(upsample)].reset_index(drop=True)  
        df['view'] = [next(views) for view in range(len(df))]
        # df = prep.get_dims(df, augment)
        
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

class NPYLoader(Dataset):
    """
    Custom Nrrd file Dataloader class for pytorch; Will utilize raw and segmented Nrrd files to select random
    regions of the image for analysis.
    """
    def __init__(self, data, method:str=None, augmentation:str=None, masksize:int=None, norms:str='norm', testing:bool=False):
        super().__init__()
        self.data = data
        self.augment = augmentation
        self.masksize = masksize
        self.norms = norms
        self.method = method
        self.testing = False

    def __len__(self)->int:
        """
        Returns length of data
        """
        return len(self.data)

    def __getslice__(self, img:np.array, thres:np.array, row:pd.DataFrame, edges:list, testing:bool=False):
        """
        returns slice of nrrd file
        -----------
        Parameters:
        --------
        Returns:
        im - np.array()
            Contains original image of size (1,64,64) 
        thres - np.array()
            Contains segmentation mask of size (1,64,64) 
        """
        im = np.zeros((1,64,64))
        mask = np.zeros((1,64,64))
        
        if edges[0] != edges[-1]:
            if testing:
                sliceid = edges[int(len(edges)/2)]
            else:
                sliceid = int(np.random.choice(np.arange(edges[0],edges[-1])))
        else:
            sliceid = edges[0]

        if row['view'] == 'x':
            im[0,:,:] += img[sliceid, :, :]
            mask[0,:,:] += thres[sliceid,:,:]

        if row['view'] == 'y':
            im[0,:,:] += img[:,sliceid, :]
            mask[0,:,:] += thres[:,sliceid,:]

        if row['view'] == 'z':
            im[0,:,:] += img[:,:,sliceid]
            mask[0,:,:] += thres[:,:,sliceid]
        
        return im, mask

    def __getitem__(self, index:int):
        row = self.data.iloc[index]
        print(row['pid'])

        img = np.load(row['uri'])
        thres = np.load(row['thresh_uri'])
        edges = prep.scan_3darray(thres, view=row['view'], threshold=1)

        img, thres = self.__getslice__(img, thres, row, edges, testing=self.testing)
        # print(thres.shape)
        if self.method != 'Original':
            sample = {'image': prep.normalize_img(prep.seg_method(img, thres, method= self.method), self.norms),
                    'label': int(row['ca']),
                    'id': row['pid']
                    }
        else:
            sample = {'image': prep.normalize_img(img, self.norms),
                'label': int(row['ca']),
                'id': row['pid']}
            
        return sample

