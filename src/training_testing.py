
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Libraries and Dependencies
# --------------------------------------------
from sklearn.metrics import roc_curve, auc, confusion_matrix
from loss import FocalLoss

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
import cv2

import numpy as np
import image
import time
# --------------------------------------------

def optim_criterion_select(net, opt):
    """
    Description:
    Input:
    Output:
    """
    # Define Loss Functions
    if opt['loss'] == 'focal':
        crit = FocalLoss().cuda()
    if opt['loss'] == 'entropy':
        crit = nn.CrossEntropyLoss().cuda()

    # Define Optimizer
    if opt['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = opt['lr'], betas= opt['betas'], eps= opt['eps'])
    if opt['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr= opt['lr'], momentum= opt['momentum'])
    if opt['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr = opt['lr'], rho = opt['rho'], eps = opt['eps'], weight_decay = opt['decay'])

    return crit, optimizer

def train(trainloader, testloader, net, device, rep, opt, model):
    """
    Description:
    Input:
    Output:
    """

    criterion, optimizer = optim_criterion_select(net, opt)

    trainLoss = np.zeros((opt['epchs']))    # Defining Zero Array of Training Loss
    validLoss = np.zeros((opt['epchs']))    # Defining Zero Array for Validation Loss
    trainAcc = np.zeros((opt['epchs']))     # Defining Zero Array for Training Accuracy
    validAcc = np.zeros((opt['epchs']))     # Defining Zero Array for Validation Accuracy
    trainTime = np.zeros((opt['epchs']))    # Defining Zero Array for Training Time

    
    for epoch in range(opt['epchs']):  # loop over the dataset multiple times

        EpochTime = 0       # Zeroing Epoch Timer
        running_loss = 0.0  # Zeroing Running Loss per epoch
        total = 0           # Zeroing total images processed
        correct = 0         # Zeroing total classes correct

        end = time.time()
        for i, (images, labels) in enumerate(trainloader):
            # Input
            images = images.to(device = device, dtype = torch.float)
            labels = labels.to(device = device)
            images = torch.autograd.Variable(images)
            labels = torch.autograd.Variable(labels)
            
            optimizer.zero_grad()
            
            output = net(images)
            _, predicted = torch.max(output,1)
            loss = criterion(output, labels)

            # Accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Gradient Descent
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        trainLoss[epoch] = running_loss/i
        trainAcc[epoch]  = (correct/total) * 100 
        trainTime[epoch] = time.time()-end
        
        validLoss[epoch], validAcc[epoch] = validate(testloader, criterion, net, device, model)
              
        running_loss = 0.0

    return trainLoss, validLoss, trainAcc, validAcc, np.mean(trainTime)

def validate(testloader, criterion, net, device, mode):
    """
    Description:
    Input:
    Output:
    """

    with torch.no_grad():
        running_loss = 0 
        total = 0
        correct = 0

        for i, (images, labels) in enumerate(testloader):

            # Load Images
            images = images.to(device = device, dtype = torch.float)
            labels = labels.to(device = device)
            input_var = torch.autograd.Variable(images)
            target_var = torch.autograd.Variable(labels)
            
            output = net(input_var)

            _, predicted = torch.max(output,1)
            loss = criterion(output, target_var)

            # Accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()
        

    return (running_loss/i, correct/total * 100)

def test(testloader, net, device, flags, mode = 3):
    correct = 0
    total = 0
    
    with torch.no_grad():
        
        targets = []        # np.zeros(len(testloader))
        prediction = []     # np.transpose(np.zeros(len(testloader)))
        softpred = []
        img_mask = []
        original_img = []
        tpos = 0
        fpos = 0
        tneg = 0
        fneg = 0
        count = 0
        
        if flags['CreateComposites']:
            FalsePos_orimg = []
            TruePos_orimg = []
            FalseNeg_orimg = []
            TrueNeg_orimg = []

        for (images, labels) in testloader:
            images_T = images.to(device = device, dtype = torch.float)
            labels = labels.to(device = device)

            outputs, x_avg = net(images_T, flag = True)                                                

            _, pred = torch.max(outputs,1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
            for i in range(len(labels)):
                img_mask.append(x_avg[i].cpu().detach().numpy())
                original_img.append(images[i].numpy())
                targets.append(labels[i].cpu().squeeze().numpy())
                prediction.append(pred[i].cpu().squeeze().numpy())
                softpred.append(outputs[i,1].cpu().squeeze().numpy())
                count += 1

                if flags['CreateComposites']:
                    ori_img = np.zeros((64,64,1))
                    ori_img[:,:,:] = images[i].numpy().T   
    
                if labels[i] == 1:
                    if labels[i] == pred[i]:
                        tpos += 1
                        if flags['CreateComposites']:
                            TruePos_orimg.append(ori_img)
                    else:
                        fpos += 1
                        if flags['CreateComposites']:
                            FalsePos_orimg.append(ori_img)

                else:
                    if labels[i] == pred[i]:
                        tneg += 1
                        if flags['CreateComposites']:
                            TrueNeg_orimg.append(ori_img)
                    else:
                        fneg += 1 
                        if flags['CreateComposites']:
                            FalseNeg_orimg.append(ori_img)
                
        sens = tpos / (tpos + fneg)
        spec = tneg / (tneg + fpos)
        acc = (100 * correct/total)
        print('Accuracy of Network: %d %%' %acc)

        fps, tps, threshold = roc_curve(targets, softpred[:])

        conf_matrix = confusion_matrix(prediction, targets)
    
    if flags['CreateComposites']:
        return (
            conf_matrix,        # Test Confusion Matrix for Malignant and Benign Nodules
            fps,                # False Positive Rates
            tps,                # True Positiive Rates
            sens,               # Sensitivity
            spec,               # Specificity
            acc,                # Accuracy of network
            TrueNeg_orimg,
            TruePos_orimg,
            FalsePos_orimg,
            FalseNeg_orimg
        )

    else:
        return( 
            conf_matrix, fps, tps,           
            sens, spec, acc)

def eval_bestmodel(net, X, y, filenames, device, dataset = None, ms = None):
    """
    Description:
    Inputs:
    Outputs:
    """
    for i in range(len(X)):
        im = np.zeros((1,1,64,64))
        im[0,:,:,:] = X[i]
        label = y[i]
        fileid = filenames[i]

        torch_im = torch.from_numpy(im)
        torch_label = torch.from_numpy(np.asarray(label))

        img = torch_im.to(device=device, dtype=torch.float)
        label = torch_label.to(device=device)
    
        output, x_avg = net(img, flag=True)

        _, pred = torch.max(output, 1)

        if ((pred == 1) and (pred == label)):
            fileid = fileid.split('.')[0] + '_TP'
            folder = '/GradCAM/TP/'
        elif ((pred == 1) and (pred != label)):
            fileid = fileid.split('.')[0] + '_FP'
            folder = '/GradCAM/FP/'
        elif ((pred == 0) and (pred == label)):
            fileid = fileid.split('.')[0] + '_TN'
            folder = '/GradCAM/TN/'
        elif ((pred == 0) and (pred != label)):
            fileid = fileid.split('.')[0] + '_FN'
            folder = '/GradCAM/FN/'
        
        x_avg = x_avg.cpu().detach().numpy()
        
        image.showImageCam(im, x_avg, net, dataset, ms = ms, title = fileid, folder = folder)