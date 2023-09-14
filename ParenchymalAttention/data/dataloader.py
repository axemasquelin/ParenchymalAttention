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
from torchvision import transforms as T
from torch.utils.data import Dataset
from torch.nn import ModuleList
from sklearn.utils import resample
from PIL import Image, ImageOps
from itertools import cycle
import pandas as pd
import numpy as np
import random
import glob
import tifffile
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
    
    if ext== '.tif': 
        original_filelist = glob.glob('./dataset/Original/*'+ ext)
        pid = [os.path.basename(x).split('_')[0] for x in original_filelist]
        ca = [os.path.basename(x).split('_')[1].split('.')[0] for x in original_filelist]
        time = [os.path.basename(x).split('_')[2].split('.')[0] for x in original_filelist]
        df = pd.DataFrame(np.transpose([pid,original_filelist,ca,time]), columns=['pid','uri', 'ca', 'time'])
        
        masked_filelist = glob.glob('./dataset/Segmented/*'+ ext)
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
        # df['view'] = [np.random.choice(['x','y','z']) for index in df.index]
        df['view'] = ['z' for index in df.index]

    else:
        views = cycle(['z'])
        # views = cycle(['x','y','z'])
        df = df.loc[df.index.repeat(upsample)].reset_index(drop=True)  
        df['view'] = [next(views) for view in range(len(df))]
        # df = prep.get_dims(df, augment)
        
    return df

class NPYLoader(Dataset):
    """
    Custom Nrrd file Dataloader class for pytorch; Will utilize raw and segmented Nrrd files to select random
    regions of the image for analysis.
    """
    def __init__(self, data, method:str=None, augmentation:str=True, masksize:int=None, norms:str='norm', testing:bool=False):
        super().__init__()
        self.data = data
        self.augment = augmentation
        self.masksize = masksize
        self.norms = norms
        self.method = method
        self.testing = testing

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
        im = np.zeros((1,img.shape[0],img.shape[1]))
        mask = np.zeros((1,thres.shape[0],thres.shape[1]))

        if edges[0] != edges[-1]:
            if testing:
                sliceid = edges[int(len(edges)/2)]
            else:
                sliceid = random.choice(edges)
        else:
            sliceid = edges[0]

        if row['view'] == 'x':
            im[0,:,:] = img[sliceid, :, :]
            mask[0,:,:] = thres[sliceid,:,:]

        if row['view'] == 'y':
            im[0,:,:] = img[:,sliceid, :]
            mask[0,:,:] = thres[:,sliceid,:]

        if row['view'] == 'z':
            im[0,:,:] = img[:,:,sliceid]
            mask[0,:,:] = thres[:,:,sliceid]
        

        return im, mask
    
    def __augment__(self, img, augment):
        """
        Randomly applies an torch augment to an image
        """

        if augment and self.testing==False:
            img = Image.fromarray(img)

            transforms = T.RandomApply(ModuleList([
                T.RandomPerspective(),
                T.RandomAffine(degrees=(0,180)),
                T.RandomRotation(degrees=(0,180)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                ]), p=0.3)
            
            img = transforms(img)

        return np.asarray(img)
    
    def __getitem__(self, index:int):
        """
        Gets Item from dataframe and return a sample containing input:image, label:diagnosis, and id:patient_id
        """
        seed =  np.random.randint(99999999)
        random.seed(seed)

        row = self.data.iloc[index]
        
        img = tifffile.imread(row['uri'])
        thres = tifffile.imread(row['thresh_uri'])
        edges = prep.scan_3darray(thres, view=row['view'], threshold=1)
        img, thres = self.__getslice__(img, thres, row, edges, testing=self.testing)
        img = prep.normalize_img(img, row['pid'], self.norms)

        if self.method != 'Original':
            img = prep.seg_method(img, thres, method= self.method)
            # img[0,:,:] = self.__augment__(img[0], augment=self.augment)
            sample = {'image': img,
                    'label': int(row['ca']),
                    'id': row['pid'],
                    'time': row['time']
                    }
        else:            
            # img[0,:,:] = self.__augment__(img[0], augment=self.augment)
            sample = {'image': img,
                'label': int(row['ca']),
                'id': row['pid'],
                'time': row['time']}
        
        return sample


