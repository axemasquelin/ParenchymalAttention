# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description: Image manipulation and visualization function
'''
# Libraries
# ---------------------------------------------------------------------------- #
# from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import os, cv2, glob, csv
# ---------------------------------------------------------------------------- #
def resaveAttention(img, image_name, folderpath):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, vmin = 150 , vmax = 255)
    plt.colorbar()
    plt.savefig(image_name)
    plt.close()

def data_load(filename, df_reference, var_names = None, clean = True):
    """
    Definition: Loads CSV Files
    Input:
    Output:
    """
    vars = []
    df = pd.read_csv(filename, low_memory = False)

    for key, val in var_names.items():
        if var_names[key] in df.columns:
            vars.append(var_names[key])

    df = df[vars]
    # df = df[df.selected != 0]

    pid_reference = df_reference.values
    df = df[df['pid'].isin(pid_reference)]

    return df

def get_imgdata(folder, filename):
    """
    Parameters:
    Returns:
    filelist - list
    """
    df_imgs = pd.DataFrame()
    filelist = []
    timepoint = []
    pid = []
    ca = []
    classification = []
    print(folder)
    filepaths = glob.glob(folder + '*_original.png')
    for filepath in filepaths:
        file = os.path.basename(filepath)
        pid.append(int(file.split('_')[0]))
        ca.append(file.split('_')[1])
        timepoint.append(int(file.split('_')[2][1]))
        classification.append(file.split('_')[-2])
    
    df_imgs['pid'] = pid
    df_imgs['study_yr'] = timepoint
    df_imgs['ca'] = ca
    df_imgs['classification'] = classification
    
    return df_imgs


def eval_MetaData(df, abnormalities = None):
    '''
    Description
    Input:  1) df_imgs          - Dataframe containing Images Metadata
            2) df_prsn          - Dataframe containing patient metadata
            3) df_sctimage      - Dataframe containing image metadata
            4) df_sctabn        - Dataframe containing abnormalities from given image
            5) abnormalities    - Abnormality defintions from NLST
    Output: 
    '''

    ## Create New Dataframe
    df_stats = pd.DataFrame()
    df_stats['Individuals'] = [len(df['pid'].values)]
    for abnormality in abnormalities:
        if abnormality !='pid':
            df_stats[abnormality + '_mean'] = [df[abnormality].mean()]
            df_stats[abnormality + '_std'] = [df[abnormality].std()]

    df_stats = df_stats.transpose()

    return df_stats

def stattests(arr:np.array, feature:str, alpha:float=0.05):
    '''
    Statistical Analysis of Features across False Positive (Index=0),
    false negative (index 1), true positive (inde=2), and true negative
    (index 3)
    -----------
    Parameters:
    arr - np.array
        contains all values for given feature
    feature - string
        name of the feature being analysed
    alpha - float
        cut of point for demonstrating method does not hold central limit theorem.     
    '''
    # FP: index 0, FN: index 1, TP: index 2, TN: index 3
    
    # False Positive - True Negative
    stat,p = scipy.stats.levene(arr[0],arr[3])
    if p < alpha:
        # print('Unequal Variance TN-FP')
        t, p_tnfp = scipy.stats.kruskal(arr[0], arr[3])
    else:
        t, p_tnfp = scipy.stats.ttest_ind(arr[0], arr[3])
    
    # False Positive - True Positive
    stat,p = scipy.stats.levene(arr[0],arr[2])
    if p < alpha:
        # print('Unequal Variance TP-FP')
        t, p_tpfp = scipy.stats.kruskal(arr[0], arr[2])
    else:
        t, p_tpfp = scipy.stats.ttest_ind(arr[0], arr[2])

    # False Negative - True Positive
    stat,p = scipy.stats.levene(arr[1],arr[2])
    if p < alpha:
        # print('Unequal Variance TP-FN')
        t, p_tpfn = scipy.stats.kruskal(arr[1], arr[2])
    else:
        t, p_tpfn = scipy.stats.ttest_ind(arr[1], arr[2])
    
    # False Negative - True Negative
    stat,p = scipy.stats.levene(arr[1],arr[3])
    if p < alpha:
        # print('Unequal Variance TP-FP')
        t, p_tnfn = scipy.stats.kruskal(arr[1], arr[3])
    else:
        t, p_tnfn = scipy.stats.ttest_ind(arr[1], arr[3])

    # False Positive - False Negative
    stat,p = scipy.stats.levene(arr[0],arr[1])
    if p < alpha:
        # print('Unequal Variance TP-TN')
        t, p_fnfp = scipy.stats.kruskal(arr[0], arr[1])
    else:
        t, p_fnfp = scipy.stats.ttest_ind(arr[0], arr[1])
       
    
    # True Positive - True Negative
    stat,p = scipy.stats.levene(arr[2],arr[3])
    if p < alpha:
        # print('Unequal Variance TP-TN')
        t, p_tntp = scipy.stats.kruskal(arr[2], arr[3])
    else:
        t, p_tntp = scipy.stats.ttest_ind(arr[2], arr[3])
    
    # print(f' {feature} TP-FP p-value: {p_tpfp}')
    # print(f' {feature} TP-FN p-value: {p_tpfn}')
    # print(f' {feature} TN-FP p-value: {p_tnfp}')
    # print(f' {feature} TN-FN p-value: {p_tnfn}')
    # print(f' {feature} TN-TP p-value: {p_tntp}')
    # print(f' {feature} FN-FP p-value: {p_fnfp}')

    return (p_tpfp, p_tpfn, p_tnfp, p_tnfn, p_tntp, p_fnfp)


def merge_dfs(df_imgs, df_prsn, df_sctabn, df_sctimage):
    """
    Parameters:
    Returns:
    """

    df = pd.merge(df_prsn, df_sctabn, on='pid')
    df = pd.merge(df, df_sctimage, on='pid')
    df = pd.merge(df, df_imgs, on=['pid','study_yr'])

    return df

def load_rads(df:pd.DataFrame, rad_file:str, abnormalities:list):
    """
    Parameters:
    df - pandas.dataframe
    rad_file - string
        radiomic csv file path
    abnormalities - list
        list of features to keep from radiomic file
    """
    df_rads = pd.read_csv(rad_file, header=0)
    df_rads = df_rads[abnormalities]
    df = pd.merge(df,df_rads,on='pid')
    
    return df



def main():

    methods = [
        'Original/',
        'Tumor-only/',
        'Parenchyma-only/'
    ]

    subfolders = [
        'FP/',
        'FN/',
        'TP/',
        'TN/',
    ]
    
    variables = {
            'pid': 'pid',
            'age': 'age',
            'pack_years': 'pkyr',
            'nodule_size': 'SCT_LONG_DIA',
            'Kilo_VP': 'kvp',
            'slice_thickness': 'reconthickness',
            'Tube_Current': 'mas',
            'attenuation': 'SCT_PRE_ATT',
            'selected': 'selected',
            'study_year': 'STUDY_YEAR',
            'study_year': 'study_yr'
    }

    Abn_features = [
        'pid',
        'xnorm',                        # X position normalized to corina
        'ynorm',                        # Y position normalized to corina
        'znorm',                        # Z position normalized to corina
        'laa950perc10b',                # Emphysema in 10mm boundary surrounding nodule
        'meanintensity0i',              # Nodule average voxel intensity (Metric of density)
        'compactness10i',               # How dense is the nodule
        'sphericity0i',                 # How round is the nodule   
        'graylevels0i',
        'energy0i',
        'entropy0i',
        'skewness0i',
        'kurtosis0i',
        'maximum3ddiameter0i',
    ]

    rad_file = '/media/axel/Linux/University of Vermont/Research/radiomics/dataset/data_Original.csv' 

    flag_cv2plt = False

    resultspath = os.getcwd() + '/results/'

    folder_cohort = os.getcwd() + '/Cohort Data/'
    files_cohort = glob.glob(folder_cohort + "/*.csv")

    for method in methods:
        diam_arr = []
        pkyr_arr = []
        age_arr = []
        xnorm_arr = []
        ynorm_arr = []
        znorm_arr = []
        laa95010b_arr = []
        meanintensity_arr = []
        compactness10i_arr = []
        sphericity0i_arr = []
        graylevels0i_arr = []
        energy0i_arr = []
        entropy0i_arr = []
        skewness0i_arr = []
        kurtosis0i_arr = []
        maximum3ddiameter0i_arr = []

        ptable = pd.DataFrame()

        for subfolder in subfolders:
            print('Evaluating %s'%(method + subfolder))
            imgfolder = resultspath + method + "GradCAM/2_9/" + subfolder
            df_imgs = get_imgdata(imgfolder, filename='*_original.png')
            print("Shape of Image Dataframe: ", df_imgs.shape)
            # df_prsn = data_load(files_cohort[0], df_imgs['pid'], var_names = variables)      # DF with Patient Demographics   
            # df_sctabn = data_load(files_cohort[1], df_imgs['pid'], var_names = variables)    # DF with CT Image Information
            # df_sctimage = data_load(files_cohort[2], df_imgs['pid'], var_names = variables)  # DF with CT Abnormalities 
            # df_prsn = df_prsn.dropna(axis=0)
            # df_sctabn = df_sctabn.dropna(axis=0)
            # df_sctimage = df_sctimage.dropna(axis=0)
            
            # df = merge_dfs(df_imgs, df_prsn, df_sctabn, df_sctimage)
            # df = df.drop_duplicates()
            
            df = load_rads(df_imgs,rad_file, abnormalities=Abn_features)
            print("Shape of Dataframe: ", df.shape)
            df_stats = eval_MetaData(df, abnormalities=Abn_features)
            
            df_stats.to_csv(resultspath + method + 'GradCAM/' + subfolder.split('/')[-2] + '_Information.csv')

            # diam_arr.append(df_sctabn['SCT_LONG_DIA'].values)
            # pkyr_arr.append(df_prsn['pkyr'].values)
            # age_arr.append(df_prsn['age'].values)
            xnorm_arr.append(df['xnorm'].values)
            ynorm_arr.append(df['ynorm'].values)
            znorm_arr.append(df['znorm'].values)
            laa95010b_arr.append(df['laa950perc10b'].values)
            meanintensity_arr.append(df['meanintensity0i'].values)
            compactness10i_arr.append(df['compactness10i'].values)
            sphericity0i_arr.append(df['sphericity0i'].values)
            graylevels0i_arr.append(df['graylevels0i'].values)
            energy0i_arr.append(df['energy0i'].values)
            entropy0i_arr.append(df['entropy0i'].values)
            skewness0i_arr.append(df['skewness0i'].values)
            kurtosis0i_arr.append(df['kurtosis0i'].values)
            maximum3ddiameter0i_arr.append(df['maximum3ddiameter0i'].values)

        ptable['Index'] = ['TP-FP','TP-FN','TN-FP','TN-FN','TN-TP','FN-FP']

        # ptable['Nodule Diameter'] = stattests(diam_arr, feature='Nodule Diameter')            
        # ptable['Pack-years'] = stattests(pkyr_arr, feature='Pack-years')
        # ptable['Age'] = stattests(age_arr, feature='Age')
        ptable['X Norm'] = stattests(xnorm_arr, feature='X Norm')
        ptable['Y Norm'] = stattests(ynorm_arr, feature='Y Norm')
        ptable['Z Norm'] = stattests(znorm_arr, feature='Z Norm')
        ptable['Laa950'] = stattests(laa95010b_arr, feature='Laa950 10mm Boundary')
        ptable['Mean Intensity'] = stattests(meanintensity_arr, feature='Mean Intensity')
        ptable['Compactness'] = stattests(compactness10i_arr, feature='Nodule Compactness')
        ptable['Sphericity'] = stattests(sphericity0i_arr, feature='Nodule Sphericity')
        ptable['Gray Levels'] = stattests(graylevels0i_arr, feature= 'Gray Levels')
        ptable['Nodule Energy'] = stattests(energy0i_arr, feature= 'Nodule Energy')
        ptable['Nodule Entropy'] = stattests(entropy0i_arr, feature= 'Nodule Entropy')
        ptable['Nodule Skewness'] = stattests(skewness0i_arr, feature= 'Nodule Skewness')
        ptable['Nodule Kurtosis'] = stattests(kurtosis0i_arr, feature= 'Nodule Kurtosis')
        ptable['Nodule Max Diameter'] = stattests(maximum3ddiameter0i_arr, feature= 'Nodule Maximum Diameter')
        ptable.T.to_csv(resultspath + method + '/' + 'Pvalues.csv')

if __name__ == '__main__':
    main()

    