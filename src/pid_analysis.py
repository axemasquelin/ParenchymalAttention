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
import os, cv2, glob
# ---------------------------------------------------------------------------- #
def resaveAttention(img, image_name, folderpath):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, vmin = 150 , vmax = 255)
    plt.colorbar()
    plt.savefig(image_name)
    plt.close()


def data_load(filename, df_reference, var_names = None, clean = True):
    """
    Definition:
    Input:
    Output:
    """
    vars = []
    df = pd.read_csv(filename, low_memory = False)

    for key, val in var_names.items():
        if var_names[key] in df.columns:
            vars.append(var_names[key])

    df = df[vars]
    df = df[df.selected != 0]
    
    pid_reference = df_reference['pid'].to_list()
    
    df = df[df['pid'].isin(pid_reference)]

    # if clean:
    #     # Drop out the columns that have over 90% missing data points.
    #     # df = df.dropna(axis=0)

    return df

def get_imagedata(imgfiles, flag_cv2plt = False):
    
    df_imgs = pd.DataFrame()
    timepoint = []
    sliceview = []
    imgname = []
    pid = []
    ca = []

    for imgfile in files:
        img = cv2.imread(imgfile)
        imgname = imgfile.split('/')[-1]
        pid.append(int(imgname.split('_')[0]))
        timepoint.append(imgname.split('_')[1])
        sliceview.append(imgname.split('_')[3])
        ca.append(imgname.split('_')[4])
        
        image_type = imgfile.split('_')[-1]
    # print(image_type)
    if flag_cv2plt:
        if image_type == 'attention.png':
            resaveAttention(img, imgfile, imgfolder)

    df_imgs['pid'] = pid
    df_imgs['yr'] = timepoint
    df_imgs['sliceview'] = sliceview
    df_imgs['ca'] = ca

    return df_imgs

def eval_MetaData(df_imgs, df_prsn, df_sctimage, df_sctabn, abnormalites = None):
    '''
    Description
    Input:  1) df_imgs          - Dataframe containing Images Metadata
            2) df_prsn          - Dataframe containing patient metadata
            3) df_sctimage      - Dataframe containing image metadata
            4) df_sctabn        - Dataframe containing abnormalities from given image
            5) abnormalities    - Abnormality defintions from NLST
    Output: 
    '''
    ## Analysing Df_imgs
    num_indv = len(df_imgs)     # Number of Individuals within Subfolder
    
    num_xslice = df_imgs['sliceview'].value_counts()['xSlice']
    num_yslice = df_imgs['sliceview'].value_counts()['ySlice']
    num_zslice = df_imgs['sliceview'].value_counts()['zSlice']

    ## Patient Demographic Information
    male_count = df_prsn['gender'].value_counts()[1]     # Number of Male Individuals
    female_count = df_prsn['gender'].value_counts()[2]   # Number of Female Individuals

    avg_age = df_prsn['age'].mean()                     # Average Individual Age
    avg_pkyr = df_prsn['pkyr'].mean()                   # Average Smoke Pack years

    avg_nodule = df_sctabn['SCT_LONG_DIA'].mean()       # Average Nodule Size


    ## Create New Dataframe
    df = pd.DataFrame({ 'Number of Individuals': [len(df_imgs)],
                        'Average Age': [df_prsn['age'].mean()],
                        'STD Age': [df_prsn['age'].std()],          
                        'Average Pkyrs': [df_prsn['pkyr'].mean()],
                        'STD Pkyrs': [df_prsn['pkyr'].std()],
                        'Average Nodule Size': [df_sctabn['SCT_LONG_DIA'].mean()],
                        'STD Nodule Size': [df_sctabn['SCT_LONG_DIA'].std()],
                        'Male Count': [df_prsn['gender'].value_counts()[1]],
                        'Female Count': [df_prsn['gender'].value_counts()[2]],
                        'Count Xslice': [df_imgs['sliceview'].value_counts()['xSlice']],
                        'Count Yslice': [df_imgs['sliceview'].value_counts()['ySlice']],
                        'Count Zslice': [df_imgs['sliceview'].value_counts()['zSlice']]
                        })

        
    ## Abnormalities From Dataset
    abn_valuecounts = df_sctabn['SCT_AB_DESC'].value_counts()

    ## Predominant Attenuation Information
    Att_valuecounts = df_sctabn['SCT_PRE_ATT'].value_counts()

    for indx in abn_valuecounts.index:
        df[abnormalites[int(indx)]] = [abn_valuecounts[indx]]

    for indx in Att_valuecounts.index:
        df[abnormalites[int(indx)]] = [Att_valuecounts[indx]]

    df = df.transpose()

    return df

def ttests(diam_arr, pkyr_arr, age_arr):
    '''
    '''

    # FP: index 0, FN: index 1, TP: index 2, TN: index 3

    t, p_tnfp = scipy.stats.ttest_ind(diam_arr[0][:], diam_arr[3][:])
    t, p_tpfn = scipy.stats.ttest_ind(diam_arr[1][:], diam_arr[2][:])

    print('Nodule Diameter TN-FP p-value: %f'%(p_tnfp))
    print('Nodule Diameter TP-FN p-value: %f'%(p_tpfn))

    t, p_tnfp = scipy.stats.ttest_ind(pkyr_arr[0][:], pkyr_arr[3][:])
    t, p_tpfn = scipy.stats.ttest_ind(pkyr_arr[1][:], pkyr_arr[2][:])

    print('Packyears TN-FP p-value: %f'%(p_tnfp))
    print('Packyears TP-FN p-value: %f'%(p_tpfn))

    t, p_tnfp = scipy.stats.ttest_ind(age_arr[0][:], age_arr[3][:])
    t, p_tpfn = scipy.stats.ttest_ind(age_arr[1][:], age_arr[2][:])
    
    print('Age TN-FP p-value: %f'%(p_tnfp))
    print('Age TP-FN p-value: %f'%(p_tpfn))

if __name__ == '__main__':
    methods = [
        'Original/',
        'OtsuMask/16x/',
        'OtsuMask/32x/',
        'OtsuMask/48x/',
        'OtsuMask/64x/',
        'BlockMask/16x/',
        'BlockMask/32x/',
        'BlockMask/48x/',
        'BlockMask/64x/',
    ]

    subfolders = [
        'GradCAM/FP/',
        'GradCAM/FN/',
        'GradCAM/TP/',
        'GradCAM/TN/',
    ]
    
    variables = {
            'pid': 'pid',
            'age': 'age',
            'pack_years': 'pkyr',
            'gender': 'gender',
            'nodule_size': 'SCT_LONG_DIA',
            'abnormalities': 'SCT_AB_DESC',
            'Kilo_VP': 'kvp',
            'slice_thickness': 'reconthickness',
            'Tube_Current': 'mas',
            'attenuation': 'SCT_PRE_ATT',
            'selected': 'selected'
    }

    Abn_features = {     #List of Abnormal Description Found in NLST dataset
        1: 'Soft Tissue',
        2: 'Ground Glass',
        3: 'Mixed',
        4: 'Fluid/Water',
        6: 'Fat Count',
        7: 'Other Tissue Count',
        9: 'Unable to Determine Count',
        51:'Non-calcified Nodule or mass (opacity >= 4mm diameter)',
        52:'Non-calcified micronodule(s) (opacity < 4mm diameter)',
        53:'Benign Lung Nodule(s)',
        54:'Atelectasis, segmental or greater',
        55:'Pleural thickening or effusion',
        56:'Non-calcified hilar/mediastinal adenopathy or mass (>= 10mm short axis)',
        57:'Chest Wall Abnormality (bone destruction, metastisis)',
        58:'Consolidation',
        59:'Emphysema',
        60:'Significant Cardiovascular abnormality',
        61:'Reticular or reticulonodular opacities, honeycombing, fibrosis',
        62:'6 or more nodules, not suspicious for cancer',
        63:'other potentially significant abnormality above diaphram',
        64:' other potentially significant abnormalities below diaphram',
        65:'Other minor abnormalities'
    }
    

    flag_cv2plt = True

    resultspath = os.path.split(os.getcwd())[0] + '/results/'

    folder_cohort = os.path.split(os.getcwd())[0] + '/Cohort Data/'
    files_cohort = glob.glob(folder_cohort + "/*.csv")
    
    for method in methods:
        diam_arr = []
        pkyr_arr = []
        age_arr = []
        for subfolder in subfolders:
            print('Evaluating %s'%(method + subfolder))
            imgfolder = resultspath + method + subfolder
            files = glob.glob(imgfolder + '*.png')
            

            df_imgs = get_imagedata(files, flag_cv2plt = flag_cv2plt)
            df_prsn = data_load(files_cohort[0], df_imgs, var_names = variables)     # DF with Patient Demographics     
            df_sctimage = data_load(files_cohort[1], df_imgs, var_names = variables)   # DF with CT Abnormalities 
            df_sctabn = data_load(files_cohort[2], df_imgs, var_names = variables) # DF with CT Image Information
            
            df = eval_MetaData(df_imgs, df_prsn, df_sctimage, df_sctabn, abnormalites = Abn_features)
            
            df.to_csv(resultspath + method + 'GradCAM/' + subfolder.split('/')[-2] + '_Information.csv')

            
            
            diam_arr.append(df_sctabn['SCT_LONG_DIA'].dropna(axis=0).values)
            pkyr_arr.append(df_prsn['pkyr'].dropna(axis=0).values)
            age_arr.append(df_prsn['age'].dropna(axis=0).values)

        ttests(diam_arr, pkyr_arr, age_arr)            