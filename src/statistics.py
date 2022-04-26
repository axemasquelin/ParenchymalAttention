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
from training_testing import *
from architectures import *
import utils

import statsmodels.stats.multitest as smt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import string
import scipy
import cv2, sys, os, csv
# ---------------------------------------------------------------------------- #

def progressInfo(method, metric, model):
        
    sys.stdout.write(
            "\r Method: {0}, Model: {1}, Metric: {2}\n".format(method, model, metric)
    )   

def multitest_stats(data1, data2):
    """
    Definition:
    Inputs:
    Outputs:
    """
    
    pvals = np.zeros(6)         # Need to change this to be combination equation C(n,r) = (n!)/(r!(n-r)!)
    tvals = np.zeros(6)
    
    t, p = scipy.stats.ttest_ind(df[df.columns[0]], df[df.columns[1]])
    pvals[0], tvals[0] = p, t
    t, p = scipy.stats.ttest_ind(df[df.columns[0]], df[df.columns[2]])
    pvals[1], tvals[1] = p, t
    t, p = scipy.stats.ttest_ind(df[df.columns[1]], df[df.columns[2]])
    pvals[2], tvals[2] = p, t
    t, p = scipy.stats.ttest_ind(df[df.columns[0]], df[df.columns[3]])
    pvals[3], tvals[3] = p, t
    t, p = scipy.stats.ttest_ind(df[df.columns[1]], df[df.columns[3]])
    pvals[4], tvals[4] = p, t
    t, p = scipy.stats.ttest_ind(df[df.columns[2]], df[df.columns[3]])
    pvals[5], tvals[5] = p, t
  
    y = smt.multipletests(pvals, alpha=0.01, method='b', is_sorted = False, returnsorted = False)
  
    print("%s - %s | P-value: %.12f"%(df.columns[0], df.columns[1], y[1][0]))    
    print("%s - %s | P-value: %.12f"%(df.columns[0], df.columns[2], y[1][1]))    
    print("%s - %s | P-value: %.12f"%(df.columns[1], df.columns[2], y[1][2]))    
    print("%s - %s | P-value: %.12f"%(df.columns[0], df.columns[3], y[1][3])) 
    print("%s - %s | P-value: %.12f"%(df.columns[1], df.columns[3], y[1][4]))
    print("%s - %s | P-value: %.12f"%(df.columns[2], df.columns[3], y[1][5]))        
    return y


def annotatefig(sig, x1, x2, y, h):
    if sig < 0.05:
        if (sig < 0.05 and sig > 0.01):
            sigtext = '*'
        elif (sig < 0.01 and sig > 0.001): 
            sigtext = '**'
        elif sig < 0.001: 
            sigtext = '***'

        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        plt.text((x1+x2)*.5, y+h, sigtext , ha='center', va='bottom', color='k')
def violin_plots(df, metric, method, key,
                 sig1 = None, sig2 = None, sig3 = None,
                 sig4 = None, sig5 = None, sig6 = None):
    """
    Definitions:
    Inputs:
    Outputs:
    """

    colors = {
                'Individual':   "BuGn",
                'Combined':     "RdBu",
             }

    titles = {
            'rf': 'Random Forest',
            'svm': 'Support Vector Machine',
            'Lasso': 'LASSO Regression'
            }

    fig, ax = plt.subplots()
    chart = sns.violinplot(data = df, cut = 0,
                           inner="quartile", fontsize = 16,
                           palette= sns.color_palette(colors[key], 8))

    chart.set_xticklabels(chart.get_xticklabels(), rotation=25, horizontalalignment='right')
    plt.xlabel("Dataset", fontsize = 14)

    if metric == 'AUC':
        plt.title(titles[method] + " " + metric.upper() + " Distribution", fontweight="bold", pad = 20)
        plt.yticks([0.5,0.6,.7,.8,.9,1], fontsize=15)
        plt.ylabel(metric.upper(), fontsize = 16)
        plt.ylim(0.45,1.07)
    else:
        plt.title(method + " " + metric.capitalize() + " Distribution", fontweight="bold")
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

    if sig4 != None:
        x1, x2 = 0, 3                           # 0i vs. Maximum Diameter
        y, h, col = 1.065, .0025, 'k'
        annotatefig(sig4, x1, x2, y, h)

    if sig5 != None:
        x1, x2 = 1, 3                           # 10b/10ib vs Maximum Diameter
        y, h, col = 1.04, .0025, 'k'
        annotatefig(sig5, x1, x2, y, h)

    if sig6 != None:
        x1, x2 = 2, 3                           # 15b/15ib vs. Maximum Diameter
        y, h, col = .981, .0025, 'k'
        annotatefig(sig6, x1, x2, y, h)

    result_dir = os.path.split(os.getcwd())[0] + "/results/"
    if key == 'individual':
        plt.savefig(result_dir + metric + "_" + method + "_Across_" + key + "_Dataset.png", bbox_inches='tight',
                    dpi=600)
    else:
        plt.savefig(result_dir + metric + "_" + method + "_Across_" + key + "_Dataset.png", bbox_inches='tight',
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
            'Masked',
            # 'Combined'
            ]

    metrics = [                    # Select Metric to Evaluate
            'auc',                 # Area under the Curve
            'sensitivity',         # Network Senstivity
            'specificity',         # Network Specificity         
            'time',
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
        

        for root, dirs, files in os.walk(os.path.split(os.getcwd())[0] + "/results/", topdown = True):
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
                            if (header == 'Obfuscated'):
                                np_obf = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                            if (header == 'Original'):
                                np_orig = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                            if (header == 'Combined'):
                                np_comb = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
        cols = df.columns.tolist()
        cols = ['Original',
                'Masked'
                # 'Combined'
                ]
        
        df = df[cols]
        print(df)
        if check_stats:
            print("Comparing Original-Obfuscated")
            sob = multitest_stats(np_orig, np_obf)
            print("Comparing Original-Combined")
            soc = multitest_stats(np_orig, np_comb)
            print("Comparing Obfuscated-Combined")
            scb = multitest_stats(np_comb, np_obf)
            
        if create_violin:
            print("Violin Plots")
            if (check_stats and metric == 'auc'):
                violin_plots(df, metric, models, sig_ob = sob, sig_oc = soc, sig_cb = scb)
            else:
                violin_plots(df, metric, models)

        if metric == 'auc':
            print("Original: ", df['Original'].mean())
            print("Masked: ", df["Masked"].mean())
            
            print("Original: ", df['Original'].std())
            print("Masked: ", df["Masked"].std())
            
