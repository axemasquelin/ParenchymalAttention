# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #
from sklearn.model_selection import train_test_split
from skorch.dataset import Dataset
from training_testing import *
from architectures import *

import matplotlib.pyplot as plt
import numpy as np
import image, utils, dataload
import cv2, sys, os
import torch
# ---------------------------------------------------------------------------- #
def progressInfo(model, method, ms = 'None'):
    """
    Definition: Function to track the model, method and mask size being evaluated at this moment
    Inputs: 1) Model    - string containing model being used
            2) Method   - string containing method being used
            3) ms       - int value describing mask size being used
    """
    sys.stdout.write("\n Model: {0}, Method: {1}, Mask-size: {2}".format(model, method, ms))

def progressBar(fold, fold_max, rep, rep_max, bar_length = 50, chr = '='):
    """
    Definition:
    Inputs: 1) value        - int showing the number of repetition left
            2) endvalue     - int maximum number of repetition
            3) bar_length   - int shows how long the bar is
            4) chr          - str character to fill bar
    """
    percent = float(rep) / rep_max
    arrow = chr * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\n\r Fold {0} of {1} | [{2}] {3}%".format(fold, fold_max, arrow + spaces, int(round(percent*100))))
    print('\n')

def GPU_init(loc):
    """
    Definition: GPU Initialization function
    Inputs: loc - 0 or 1 depending on which GPU is being utilized
    Outputs: check_gpu - gpu enabled variable
    """
    check_gpu = torch.device("cuda:" + str(loc) if torch.cuda.is_available() else "cpu")
    print("Available Device: " + str(check_gpu))
    
    return check_gpu

def net_select(model):
    """
    Description: Network selection function
    Input: model - string that defines which model will be used
    Output: net - loaded network
    """
    if (model == 'MobileNet1'):
        net = MobileNetV1()
        net.init_weights()
    
    if (model == 'MobileNet2'):
        net = MobileNetV2()
        net.init_weights()
    
    if (model == "Miniception"):
        net = Miniception()
        net.init_weights()

    else:
        print("Warning: Model Not Found")
    
    return net

def eval(X,y,names,
        class_names, device,
        folds, reps, opt, flags,
        method = 'Original',
        masksize = None,
        model = 'MobileNet1', 
        seed = 2019,
        fig = 3):
    '''
    '''

    # Define Static figure counters
    static_fig = 0
    
    # Defining empty lists to store network performance information
    trainloss, valloss =  [], []
    trainacc, valacc =  [], []
    sensitivity =  np.zeros((folds,reps))
    specificity =  np.zeros((folds,reps))
    auc_scores = np.zeros((folds,reps))
    train_time = np.zeros((folds,reps))
    
    best_acc = 0
    
    for k in range(folds):

        X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(X,y,names, test_size = 0.25, random_state = k)
                
        fprs, tprs = [], []

        for r in range(reps):

            #Initialize Net
            net = net_select(model)
            net = net.to(device)
            
            if flags['CheckParams']:
                pytorch_total_params = sum(p.numel() for p in net.parameters())
                print("Number of Parameters: %i"%(pytorch_total_params))

            progressInfo(model, method, masksize)
            progressBar(k, folds, r + 1, reps)

            # Load Training Dataset
            trainset = Dataset(X_train, y_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size= 125, shuffle= True)

            #Load testing Dataset
            testset = Dataset(X_test, y_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size= 125, shuffle= True)

            trainLoss, validLoss, trainAcc, validAcc, trainTime = train(trainloader, testloader, net, device, r,  opt, method)
            
            trainloss.append(trainLoss)
            valloss.append(validLoss)
            trainacc.append(trainAcc)
            valacc.append(validAcc)
            train_time[k,r] = trainTime

            utils.plot_losses(fig, trainLoss, validLoss, method, ms = masksize)
            utils.plot_accuracies(fig, trainAcc, validAcc, method, ms = masksize)
                     
            confmatrix, fp, tp, sens, spec, acc = test(testloader, net, device, flags, mode = method)             

            if acc > best_acc:
                best_acc = acc
                utils.model_save(method, masksize, net)

                if flags['CreateComposites']: 
                    confmatrix, fp, tp, sens, spec, acc, TNoimg, TPoimg, FNoimg, FPoimg = test(testloader, net, device, flags, mode = method)    
                    
                    utils.saveAttentionImg(TNoimg, method, masksize, title= 'AverageTrueNegative_Image')
                    utils.saveAttentionImg(TPoimg, method, masksize, title= 'AverageTruePositive_Image')
                    utils.saveAttentionImg(FNoimg, method, masksize, title= 'AverageFalseNegative_Image')
                    utils.saveAttentionImg(FPoimg, method, masksize, title= 'AverageFalsePositive_Image')

                plt.figure(fig)
                utils.plot_confusion_matrix(confmatrix, class_names, r, method, masksize, normalize = True, saveFlag = True)
                fig += 1

                if flags['ImageStandard']:
                    selected_images = [
                                    '100463_T0_smooth_xSlice_0.jpg',  # Benign Nodule
                                    '100397_T1_smooth_xSlice_0.jpg',  # Benign Nodule + Parenchymal abnormalities
                                    '101606_T0_smooth_xSlice_0.jpg',  # Benign Nodule + Plural Wall
                                    '100012_T1_smooth_xSlice_1.jpg',  # Malignant Nodule
                                    '100570_T0_smooth_ySlice_1.jpg',  # Malignant Nodule + Parenchymal abnormalities
                                    '101692_T2_smooth_ySlice_1.jpg',  # Malignant Nodule + Plural Wall
                    ]
                    for img in selected_images:
                        imfile = os.path.split(os.getcwd())[0] + '/dataset/' + img
                        image.getcam(imfile, masksize, net, method, device, fileid = img)

                if flags['EvalBestModel']:
                    best_net = net
                    testname_bestacc = name_test


            sensitivity[k,r] = sens
            specificity[k,r] = spec

            fprs.append(fp), tprs.append(tp)
        
        auc_scores[k,:] = utils.calcAuc(fprs,tprs, method, masksize, fig, plot_roc= True)
        fig += 1
    
    if flags['EvalBestModel']:
        if method != 'Original':
            net = torch.load(os.path.split(os.getcwd())[0] + '/results/' + method + '/' + str(masksize) + 'x/' + method + '_bestnetwork.pt')
        else:
            net = torch.load(os.path.split(os.getcwd())[0] + '/results/' + method + '/' + method + '_bestnetwork.pt')
        for filepath in testname_bestacc:
            img_name = filepath.split('/')[-1]
            image.getcam(filepath, masksize, best_net, method, device, fileid = img_name, folder = '/GradCAM/')

    ''' Move this to a new function'''
    utils.csv_save(method, masksize, sensitivity, name = 'sensitivity')
    utils.csv_save(method, masksize, specificity, name = 'specificity')
    utils.csv_save(method, masksize, auc_scores, name = 'auc')             


def main():

    # Network Parameters

    methods = [
            'Original',
            'OtsuMask',
            'BlockMask',
            # 'Random_Otsu',
            # 'RandomBlock'
            ]

    masksizes = [
                16,
                32,
                48,
                64
    ]    

    model = 'Miniception'
    # model = 'MobileNet1'
    # model = 'MobileNet2' 

    opt = {
            'loss': 'entropy',          # entropy or focal
            'optimizer': 'Adam',        # SGD, Adam, or Adadelta
            'epchs': 125,               # Number of Epochs
            'lr': 0.0001,               # Learning Rate
            'betas': (0.9, 0.999),      # Beta parameters for Adam optimizer
            'rho': 0.9,                 # rho parameter for adadelta optimizer
            'eps': 1e-7,                # epsilon paremeter for Adam and Adadelta optimzier
            'decay': 0.001,             # Decay rate for Adadelta optimizer
            'momentum': 0.99            # Momentum parameter for SGD optimizer
        }
    
    # Flags
    flags = {
            'CreateComposites':   False,      # Create Composite Images of Saliency Maps and Original images based on TP, FP, TN,
            'EvalBestModel':      True,       # Evaluates Best Performing Model and Saves Attention Maps of test set.
            'ImageStandard':      True,      # Create Salience Maps for Standardized Image Set
            'CheckFeatures':      False,      # Check Features from Convolutional Layers
            'CheckConcepts':      False,      # Check Concept Activation Vectors (NOT Implemented)
            'CheckParams':        True,       # Check Number of Parameters in
            'SaveFigs':           True,       # Save Figure Flag
            'SaveAUC':            True,       # Save AUC Flag
    }

    # Variables
    class_names = ["Benign", "Malignant"]   # Class Name (1 - Malignant, 0 - Benign)
    gpu_loc = 0                             # Define GPU to use (0 or 1)
    seed = 2020                             # Define Random Seed
    folds = 5                               # Cross Validation Folds
    reps = 5                               # Define number of repetition for each test

    # GPU Initialization
    device = GPU_init(gpu_loc)

    # Defining Dataset Path
    cwd = os.getcwd()
    allfiles = os.path.split(cwd)[0] + '/dataset/'
    
    for method in methods:
            
        if method == 'Original':
            # Load Data
            X, y, names = dataload.load_img(allfiles, masktype = method, seed = seed)
            os.chdir(cwd)
            eval(X,y, names, class_names, device, folds, reps, opt, flags, model = model, method = method)
        
        else:
            for masksize in masksizes:
                # Load Data
                X, y, names = dataload.load_img(allfiles, masktype= method, mask_size= masksize, seed = seed)
                os.chdir(cwd)
                eval(X,y, names, class_names, device, folds, reps, opt, flags, model = model, method = method, masksize = masksize)
                        
     
if __name__ == '__main__':
    """
    Definition:
    """
    main()
    