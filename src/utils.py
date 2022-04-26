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
import cv2, os
# ---------------------------------------------------------------------------- #
def plot_losses(fig, trainLoss, validLoss, mode, ms = None):
    '''
    Description: Function Plots Loss over epochs for all methodologies
    Inputs:
        - fig: Figure Value to avoid overwritting open figures
        - trainloss: Trainloss of latest trained netowrk
        - validLoss: Validation loss of latest trained network
        - mode: Data type being processed (original, masked)
        - ms: mask size for masked dataset
    '''
    plt.figure(fig)
    plt.plot(trainLoss)
    plt.plot(validLoss)
    plt.title('Loss over Epochs ' + str(mode), fontsize = 18)
    plt.xlabel('Epochs',fontsize = 18)
    plt.ylabel('Loss (Mean Error)',fontsize = 18)
    plt.legend(['Training','Validation'], loc = 'upper right', fontsize = 18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if mode != 'Original':
        plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '/' + str(ms) + 'x' + '/' + str(mode) + str(ms) + 'x' + '_Loss.png', dpi = 600)
    else:
        plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '/' + str(mode) + '_Loss.png', dpi = 600)
    
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
    plt.figure(fig)
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

    if dataset != 'Original':
        plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(dataset) + '/' + str(ms) + 'x' + '/' + str(dataset) + str(ms) + 'x' + '_Loss.png', dpi = 600)
    else:
        plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(dataset) + '/' + str(dataset) + '_Loss.png', dpi = 600)
    
    plt.close()


    savepath = os.path.split(os.getcwd())[0] + '/results/All Approaches' + title + '.png'
    plt.savefig(savepath, dpi = 600)

    plt.close(fig)
    return fig

def plot_accuracies(fig, trainAcc, validAcc, mode, ms = None):
    '''
    Description: Function Plots Accuracies over epochs for all methodologies
    Inputs:
        - fig: Figure Value to avoid overwritting open figures
        - trainloss: Train Accuracy of latest trained netowrk
        - validLoss: Validation Accuracy of latest trained network
        - mode: Data type being processed (original, masked)
        - ms: mask size for masked dataset
    '''

    plt.figure(fig)
    plt.plot(trainAcc)
    plt.plot(validAcc)
    plt.title('Accuracies over Epochs ' + str(mode), fontsize = 20)
    plt.xlabel('Epochs', fontsize = 18)
    plt.ylabel('Accuracy', fontsize = 18)
    plt.legend(['Training','Validation'], loc = 'upper right', fontsize = 18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    if mode != 'Original':
        plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '/' + str(ms) + 'x' + '/' + str(mode) + str(ms) + 'x' + '_Accuracies.png', dpi = 600)
    else:
        plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '/' + str(mode) + '_Accuracies.png', dpi = 600)
    plt.close()

def calcAuc (fps, tps, mode, ms, reps, plot_roc = False):
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
    print(len(aucs))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if plot_roc:
        plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc, reps, mode, ms)
    print(aucs)
    return aucs


def plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc, reps, mode, ms):
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
    plt.title('ROC Curve for ' + str(mode), fontsize=20)
    plt.legend(loc="lower right", fontsize=18)

    if mode != 'Original':
        plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '/' + str(ms) + 'x' + '/' + str(mode) + str(ms) + 'x' + '_ROC.png', dpi = 600)
    else:
        plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '/' + str(mode) + '_ROC.png', dpi = 600)
    
    plt.close()


def plot_confusion_matrix(cm, classes, r, dataset, ms,
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
        if dataset != 'Original':
            title = 'Normalize Confusion Matrix ' + str(dataset) + '_' + str(ms) + 'x'
        else:
            title = 'Normalize Confusion Matrix ' + str(dataset)
        
    if saveFlag:
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        if dataset != 'Original':
            savepath = os.path.split(os.getcwd())[0] + '/results/' + str(dataset)+ '/' + str(ms) + 'x' + '/' + str(dataset)  + '_' + str(ms) + 'x' + "_best_model.png"
        else:
            savepath = os.path.split(os.getcwd())[0] + '/results/' + str(dataset) + '/' + str(dataset) + "_best_model.png"

        plt.savefig(savepath, dpi = 600)
        plt.close()
  


def calcLoss_stats(loss, mode, static_fig, figure,
                   plot_loss = True,
                   plot_static = False):

    losses = []
    
    for itr, _loss in enumerate(loss):
        print("_Loss: " + str(_loss))
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

def model_save(method, ms, net):
    '''
    Description: Function to save best performing network to result directory
    Inputs:
        - method: Original Dataset, Masked Dataset, or Combined Dataset
        - ms: Mask Size applied to remove the nodules.
        - net: Trained Network
    '''
    print("Saving Network")

    if method != 'Original':
        net_path = os.path.split(os.getcwd())[0] + "/results/" + method + '/' + str(ms) + 'x' + '/' + method + '_bestnetwork.pt'
    else:
        net_path = os.path.split(os.getcwd())[0] + "/results/" + method + '/' + method + '_bestnetwork.pt'
    torch.save(net, net_path)

def saveAttentionImg(img_list, method, ms, heatmap = False, title = None):
    '''
    Description:
    Inputs:
    Outputs:
    '''
    w = 64 #img_list[0].shape[0]
    h = 64 #img_list[0].shape[1]
    d = 3 #img_list[0].shape[2]
    img = np.zeros((w,h,d))

    if len(img_list) != 0:
        for i in range(len(img_list)):
            img += img_list[i]

        img = img/i    
        if heatmap:
            if method != 'Original':
                pth_to_save = os.path.split(os.getcwd())[0]+ "/results/" + method + '/' + str(ms) + 'x' + '/GradCAM/' + title + '.png'
            else:
                pth_to_save = os.path.split(os.getcwd())[0] + "/results/" + method + '/GradCAM/' + title + '.png'
        else:
            if method != 'Original':
                pth_to_save = os.path.split(os.getcwd())[0]+ "/results/" + method + '/' + str(ms) + 'x' + '/Composites/' + title + '.png'
            else:
                pth_to_save = os.path.split(os.getcwd())[0] + "/results/" + method + '/Composites/' + title + '.png'

        cv2.imwrite(pth_to_save, img)

def csv_save(method, ms, data, name = ''):
    ''' Save AUCs scores to a csv file '''

    cols = [name +str(i+1) for i in range(data.shape[1])]
    logs = pd.DataFrame(data, columns=cols)    

    if method != 'Original':
        pth_to_save = os.path.split(os.getcwd())[0] + "/results/" + method + '/' + str(ms) + 'x' + '/' + method + str(ms) + 'x'+  "_" + name + ".csv"
    else:
        pth_to_save = os.path.split(os.getcwd())[0] + "/results/" + method + '/' + '/' + method + "_" + name + ".csv"

    logs.to_csv(pth_to_save)

    print(logs)