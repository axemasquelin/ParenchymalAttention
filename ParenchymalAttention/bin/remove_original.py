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
from torch.nn import Softmax

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import random
import math
import time
import os, glob
# ---------------------------------------------------------------------------- #


wd = os.getcwd() + '/ParenchymalAttention/results/'
methods = ['Original', 'Parenchyma-only', 'Tumor-only']
subfolders = ['FN', 'FP', 'TN', 'TP']

for method in methods:
    for subfolder in subfolders:
        filelist = glob.glob(wd + method + '/GradCAM/2_9/' + subfolder + '/*.png')
        original_files = [os.remove(x) for x in filelist if x.endswith('_original.png')]
        