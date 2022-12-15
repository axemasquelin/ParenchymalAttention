# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description: Utility functions to plot metrics
'''
# Libraries
# ---------------------------------------------------------------------------- #
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch.utils import data

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import random
import torch
import math
import time
import sys
import cv2
import os
# ---------------------------------------------------------------------------- #

def check_parameters(netlist, params=None):
    """
    Parameters:
    -----------
    """
    sys.stdout.write('\n {0}| Number of Parameters in Networks |{0}'.format('-'*6))
    for i, net in enumerate(netlist):

        pytorch_total_params = sum(p.numel() for p in net.parameters())
        sys.stdout.write("\n {0} Number of Parameters: {1}".format(params['model'][i], pytorch_total_params))

    sys.stdout.write('\n {0}'.format('-'*48))

def csv_save(method, ms, data, name = ''):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """ 

    cols = [name +str(i+1) for i in range(data.shape[1])]
    logs = pd.DataFrame(data, columns=cols)    

    if ms != None:
        pth_to_save = os.getcwd() + "/results/" + method + '/' + str(ms) + 'x' + '/' + method + str(ms) + 'x'+  "_" + name + ".csv"
    else:
        pth_to_save = os.getcwd() + "/results/" + method + '/' + '/' + method + "_" + name + ".csv"

    logs.to_csv(pth_to_save)

    print(logs)

def model_save(method, ms, net):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """ 
    print("Saving Network")
    if ms != None:
        net_path = os.getcwd() + "/results/" + method + '/' + str(ms) + 'x' + '/' + method + '_bestnetwork.pt'
    else:
        net_path = os.getcwd() + "/results/" + method + '/' + method + '_bestnetwork.pt'
    torch.save(net, net_path)

def saveAttentionImg(img_list, method, ms, heatmap = False, title = None):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """ 
    w = 64 #img_list[0].shape[0]
    h = 64 #img_list[0].shape[1]
    d = 4 #img_list[0].shape[2]
    img = np.zeros((w,h,d))

    if len(img_list) != 0:
        for i in range(len(img_list)):
            img += img_list[i]

        img = img/i    
        if heatmap:
            if method == 'Otsu' or method == 'DropBlock':
                pth_to_save = os.getcwd()+ "/results/" + method + '/' + str(ms) + 'x' + '/GradCAM/' + title + '.png'
            else:
                pth_to_save = os.getcwd()+ "/results/" + method + '/GradCAM/' + title + '.png'
        else:
            if method == 'Otsu' or method == 'DropBlock':
                pth_to_save = os.getcwd()+ "/results/" + method + '/' + str(ms) + 'x' + '/Composites/' + title + '.png'
            else:
                pth_to_save = os.getcwd()+ "/results/" + method + '/Composites/' + title + '.png'

        cv2.imwrite(pth_to_save, img*255)

        
def create_directories(config):
    """
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """ 
    sys.stdout.write('\n\r {0} | Checking for Result Directories | {1}\n '.format('-'*25, '-'*25))
    
    for i in range(len(config['experiment_params']['maskratios'])):
        maskratio = config['experiment_params']['maskratios'][i]
        savedirectory = config['training_data']['cwd'] + '/results/' + str(maskratio) + 'x/'
        dir_exists = os.path.exists(savedirectory)
        if not dir_exists:
            sys.stdout.write('\n\r {0} | Creating Result Directories | {0}\n '.format('-'*50))
            os.mkdir(savedirectory)

