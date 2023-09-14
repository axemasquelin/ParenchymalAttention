# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description: Secondary Analysis showing the distribution of performance across all runs
'''
# Libraries
# ---------------------------------------------------------------------------- #
from sklearn.metrics import roc_curve, auc, confusion_matrix
# from statsmodels import stats

import statsmodels.stats.multitest as smt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import string
import scipy
import cv2, sys, os, csv
# ---------------------------------------------------------------------------- #

def annotatefig(sig, x1, x2, y, h):
    print(sig)
    if sig < 0.05:
        if (sig < 0.05 and sig > 0.01):
            sigtext = '*'
        elif (sig < 0.01 and sig > 0.001): 
            sigtext = '**'
        elif sig < 0.001: 
            sigtext = '***'

        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        plt.text((x1+x2)*.5, y+h, sigtext , ha='center', va='bottom', color='k')

def violin_plots(df, metric, methods,
                 sig1 = None, sig2 = None, sig3 = None,
                 sig4 = None, sig5 = None, sig6 = None):
    """
    Definitions:
    Inputs:
    Outputs:
    """

    colors = {
                'Individual':   "BuGn",
    #             'Combined':     "RdBu",
             }

    fig, ax = plt.subplots()
    chart = sns.violinplot(data = df, cut = 0,
                           inner="quartile", fontsize = 16,
                           palette= sns.color_palette(colors['Individual'], 8))

    chart.set_xticklabels(chart.get_xticklabels(), rotation=25, horizontalalignment='right')
    plt.xlabel("Dataset", fontsize = 14)

    if metric == 'AUC':
        plt.title(metric.upper() + " Distribution", fontweight="bold", pad = 20)
        plt.yticks([0.5,0.6,.7,.8,.9,1], fontsize=15)
        plt.ylabel(metric.upper(), fontsize = 16)
        plt.ylim(0.45,1.07)
    else:
        plt.title(metric.capitalize() + " Distribution", fontweight="bold")
        plt.ylabel(metric.capitalize(), fontsize = 16)
        
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if sig1 != None:
        x1, x2 = 0, 1                           # 0i vs. 10b/10ib
        y, h, col = .985, .0025, 'k'
        annotatefig(sig1, x1, x2, y, h)

    if sig2 != None:
        x1, x2 = 0, 2                           # 0i vs. 15b/15ib
        y, h, col = 1.015, .0025, 'k'
        annotatefig(sig2, x1, x2, y, h)

    if sig3 != None:
        x1, x2 = 1, 2                            # 10b/10ib vs 15b/15ib
        y, h, col = 0.993, .0025, 'k'
        annotatefig(sig3, x1, x2, y, h)

    result_dir = os.getcwd() + "/results/"
    
    plt.savefig(result_dir + metric + "_Across_Dataset.png", bbox_inches='tight',
                dpi=600)

    plt.close()

if __name__ == '__main__':
    """
    Definition:
    Inputs:
    Outputs:
    """

    # Network Parameters
    models = [
            'Original',
            'Surround-Segmentation',
            'Tumor-Segmentation'
            ]

    metrics = [                    # Select Metric to Evaluate
            'auc',                 # Area under the Curve
            'sensitivity',         # Network Senstivity
            'specificity',         # Network Specificity         
            ]
    
    # Variable Flags
    create_violin = True
    check_stats = True
    print(os.path.split(os.getcwd()))
    
    for metric in metrics:
        print(metric)
        # Dataframe Inits_  
        df = pd.DataFrame()                # General Dataframe to generate Bar-graph data
        np_obf = np.zeros((5,5))           # Conv Layer Dataframe for violin plots
        np_orig = np.zeros((5,5))          # Wavelet Layer Dataframe for violin plots
        np_comb = np.zeros((5,5))          # Multi-level Wavelet Dataframe
        
        for model in models:
            for root, dirs, files in os.walk(os.getcwd() + "/results/" +model + '/', topdown = True):
                for name in files:
                    if (name.endswith(metric + ".csv")):
                        header = name.split('_' + metric)[0]
                        if header in models:
                            mean_ = []
                            filename = os.path.join(root,name)
                            with open(filename, 'r') as f:
                                reader = csv.reader(f)
                                next(reader)
                                for row in reader:
                                    for l in range(len(row)-1):
                                        mean_.append(float(row[l+1]))
                            df[header] = np.transpose(mean_)
                            print(header)
                            if metric == 'auc':
                                if (header == 'Original'):
                                    np_orig = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                                if (header == 'Surround-Segmentation'):
                                    np_surround = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                                if (header == 'Tumor-Segmentation'):
                                    np_tumor = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
        df = df.rename({'Surround-Segmentation': 'Parenchyma-Only', 'Tumor-Segmentation': 'Tumor-Only'}, axis = 'columns')      
        cols = df.columns.tolist()
    
        
        df = df[cols]
        print(df)
        if check_stats:
            sigs = multitest_stats(np_orig, np_surround)
            print(sigs)
            
        if create_violin:
            print("Violin Plots")
            if (check_stats and metric == 'auc'):
                violin_plots(df, metric, models, sig1=sigs[1][0], sig2=sigs[1][1], sig3=sigs[1][2])
            else:
                violin_plots(df, metric, models)

        # if metric == 'auc':
        print("Original Mean: ", df['Original'].mean())
        print("Tumor-Only Mean: ", df["Tumor-Only"].mean())
        print("Parenchyma-Only Mean: ", df["Parenchyma-Only"].mean())
        
        print("Original STD: ", df['Original'].std())
        print("Tumor-Only STD: ", df["Tumor-Only"].std())
        print("Parenchyma-Only STD: ", df["Parenchyma-Only"].std())