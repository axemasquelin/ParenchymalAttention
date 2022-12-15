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

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import random
import math
import time
import os
# ---------------------------------------------------------------------------- #

def plot_metric(params):
    '''
    Description: Function Plots Accuracies over epochs for all methodologies
    Inputs:
        - fig: Figure Value to avoid overwritting open figures
        - trainloss: Train Accuracy of latest trained netowrk
        - validLoss: Validation Accuracy of latest trained network
        - mode: Data type being processed (original, masked)
        - ms: mask size for masked dataset
    '''
    plt.figure()
    plt.plot(params['trainmetric'])
    plt.plot(params['valmetric'])
    plt.title(params['title'], fontsize = 20)
    plt.xlabel(params['xlabel'], fontsize = 18)
    plt.ylabel(params['ylabel'], fontsize = 18)
    plt.legend(params['legend'], loc = 'upper right', fontsize = 18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    if params['maskratio'] != None:
        plt.savefig(os.getcwd() + '/results/' + str(params['maskratio']) + 'x/' + params['savename']+'.png', dpi = 600)
    else:
        plt.savefig(os.getcwd() + '/results/' + params['method'] + '/' + params['savename']+'.png', dpi = 600)
    
    plt.close()

def plot_MeanLoss(arr, dataset, static_fig, fig, ms,
                  xlabel = None,
                  ylabel = None,
                  title = None):
    '''
    Description: Function Plots Mean Loss over epochs for all methodologies
    Inputs:
        - arr: array containing all data for training or validation loss
        - dataset: string array describing which approach is being evaluated
        - static_fig: Static figure value (Used for calcLoss_stat function)
        - fig: counter for number of figure generated
        - xlabel: X-axis title
        - ylabel: Y-axis title
        - titel: figure title
    Output:
        - fig: Counter for number of figure generated
    '''

    mean_, _upper, _lower = calcLoss_stats(arr, dataset, static_fig, fig, plot_loss = True, plot_static= True)
    fig += 1
    plt.figure()
    plt.plot(mean_)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(dataset, loc = 'upper right')

    plt.fill_between(
        np.arange(0,opt['epchs']), _lower, _upper,
        alpha=.2, label=r'$\pm$ std.'
    )
    if ms != None:
        plt.savefig(os.getcwd() + '/results/' + str(ms) + 'x/Loss.png', dpi = 600)
    else:
        plt.savefig(os.getcwd() + '/results/' + '/Loss.png', dpi = 600)
    plt.close()

    return fig



def calcAuc (fps, tps, method, ms, reps, plot_roc = False):
    ''' Calculate mean ROC/AUC for a given set of 
        true positives (tps) & false positives (fps)
    '''

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    for itr, (_fp, _tp) in enumerate(zip(fps, tps)):
        tprs.append(np.interp(mean_fpr, _fp, _tp))
        tprs[-1][0] = 0.0
        roc_auc = auc(_fp, _tp)
        aucs.append(roc_auc)

        if plot_roc:
            plt.figure(reps, figsize=(10,8))
            plt.plot(
                _fp, _tp, lw=1, alpha=0.5,
                # label='ROC fold %d (AUC = %0.2f)' % (itr+1, roc_auc)
            )
    # print(len(aucs))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if plot_roc:
        plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc, reps, method, ms)
    # print(aucs)
    return aucs


def plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc, reps,method, ms):
    ''' Plot roc curve per fold and mean/std score of all runs '''

    plt.figure(reps, figsize=(10,8))

    plt.plot(
        mean_fpr, mean_tpr, color='k',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc)
    )

    # plot std
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        color='grey', alpha=.4, label=r'$\pm$ std.'
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('ROC Curve for Mask-ratio: ' + str(ms), fontsize=20)
    plt.legend(loc="lower right", fontsize=18)

    if ms != None:
        plt.savefig(os.getcwd() + '/results/' + '/' + method +'/'+ str(ms) + 'x/' + 'ROC_' + str(ms) + 'x.png', dpi = 600)
    else:
        plt.savefig(os.getcwd() + '/results/' + method +  '/' + method+ '_ROC.png', dpi = 600)    
    plt.close()


def plot_confusion_matrix(cm, classes, r, ms, method,
                          normalize=False,
                          title= None,
                          saveFlag = False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    tick_marks = np.arange(len(classes))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
  
    plt.colorbar()

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    if title == None:

        title = 'Normalize Confusion Matrix ' + '_' + str(ms) + 'x'
        
    if saveFlag:
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        if ms != None:
            savepath = os.getcwd() + '/results/' + method + '/' + str(ms) + 'x/' +"ConfusionMatrix_best_model.png"
        else:
            savepath = os.getcwd() + '/results/' + method + '/' +"ConfusionMatrix_best_model.png"

        plt.savefig(savepath, dpi = 600)
        plt.close()
  
def calcLoss_stats(loss, mode, static_fig, figure,
                   plot_loss = True,
                   plot_static = False):

    losses = []
    
    for itr, _loss in enumerate(loss):
        # print("_Loss: " + str(_loss))
        losses.append(_loss)
        
        if plot_loss == True:
            plt.figure(figure, figsize=(10,8))
            plt.plot(
                _loss, lw=1, alpha=0.5,
                label='Loss iteration %d' % (itr+1)
            )
        if plot_static == True:
            plt.figure(static_fig, figsize=(10,8))
            plt.plot(
                _loss, lw=1, alpha=0.5,
                label='Loss iteration %d' % (itr+1)
            )

    mean_loss = np.mean(losses, axis=0)
    std_loss = np.std(losses, axis=0)
    loss_upper = np.minimum(mean_loss + std_loss, 1)
    loss_lower = np.maximum(mean_loss - std_loss, 0)

    if plot_loss == True:
        plt.figure(figure)
        plt.plot(
            mean_loss, color='k',
            label=r'Mean Loss'
            )
        plt.fill_between(
            np.arange(0,len(mean_loss)), loss_lower, loss_upper,
            alpha=.2, label=r'$\pm$ std.'
            )
        
        plt.title(" Loss over Epochs - " + str(mode), fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(loc="upper right", fontsize=16)

    if plot_static == True:
        plt.figure(static_fig)
        plt.fill_between(
            np.arange(0,len(mean_loss)), loss_lower, loss_upper,
            alpha=.3, label=r'$\pm$ std.'
        )
        plt.title(" Loss over Epochs - All Approaches" , fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(loc="upper right", fontsize=16)

    return mean_loss, loss_upper, loss_lower
