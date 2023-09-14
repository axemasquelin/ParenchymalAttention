
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Libraries and Dependencies
# --------------------------------------------
from sklearn.metrics import roc_curve, auc, confusion_matrix

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
import cv2

import numpy as np
import ParenchymalAttention.utils.image as image
import time
# --------------------------------------------

def optim_criterion_select(net, opt):
    """
    Description:
    Input:
    Output:
    """
    # Define Loss Functions
    if opt['loss'] == 'entropy':
        crit = nn.CrossEntropyLoss().cuda()

    # Define Optimizer
    if opt['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = opt['lr'], betas= opt['betas'], eps= opt['eps'])
    if opt['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr = opt['lr'], betas= opt['betas'], eps= opt['eps'], weight_decay=0.01)
    if opt['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr= opt['lr'], momentum= opt['momentum'])
    if opt['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr = opt['lr'], rho = opt['rho'], eps = opt['eps'], weight_decay = opt['decay'])

    return crit, optimizer

def train(trainloader, valloader, net, progressbar, config):
    """
    Description:
    Input:
    Output:
    """

    criterion, optimizer = optim_criterion_select(net, config['opt'])

    trainLoss = np.zeros((config['opt']['epochs']))    # Defining Zero Array of Training Loss
    validLoss = np.zeros((config['opt']['epochs']))    # Defining Zero Array for Validation Loss
    trainAcc = np.zeros((config['opt']['epochs']))     # Defining Zero Array for Training Accuracy
    validAcc = np.zeros((config['opt']['epochs']))     # Defining Zero Array for Validation Accuracy
    trainTime = np.zeros((config['opt']['epochs']))    # Defining Zero Array for Training Time

    
    for epoch in range(config['opt']['epochs']):  # loop over the dataset multiple times

        running_loss = 0.0  # Zeroing Running Loss per epoch
        total = 0           # Zeroing total images processed
        correct = 0         # Zeroing total classes correct

        end = time.time()
        for i, data in enumerate(trainloader):
            progressbar.visual(epoch= epoch, batch_id= i, max_batches= len(trainloader), max_epoch= config['opt']['epochs'])

            # Input
            images = data['image'].to(device = config['device'], dtype = torch.float)
            labels = data['label'].to(device = config['device'])

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
        
        validLoss[epoch], validAcc[epoch] = validate(valloader, criterion, net, config['device'])
          
    return {'Trainingloss': trainLoss,
            'Validationloss': validLoss,
            'TrainingAccuracy':trainAcc,
            'ValidationAccuracy':validAcc,
            }

def validate(testloader, criterion, net, device):
    """
    Description:
    Input:
    Output:
    """

    with torch.no_grad():
        running_loss = 0 
        total = 0
        correct = 0

        for i, data in enumerate(testloader):

            # Load Images
            images = data['image'].to(device = device, dtype = torch.float)
            labels = data['label'].to(device = device)
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

def test(testloader, net, device):
    """
    Description:
    Input:
    Output:
    """
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
        
        for data in testloader:
            images = data['image'].to(device = device, dtype = torch.float)
            labels = data['label'].to(device = device)

            outputs, x_avg = net(images, flag = True)                                                

            _, pred = torch.max(outputs,1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
            for i in range(len(labels)):
                img_mask.append(x_avg[i].cpu().detach().numpy())
                original_img.append(images[i].cpu().numpy())
                targets.append(labels[i].cpu().squeeze().numpy())
                prediction.append(pred[i].cpu().squeeze().numpy())
                softpred.append(outputs[i,1].cpu().squeeze().numpy())
                count += 1

                if labels[i] == 1:
                    if labels[i] == pred[i]:
                        tpos += 1
                    else:
                        fpos += 1
                else:
                    if labels[i] == pred[i]:
                        tneg += 1

                    else:
                        fneg += 1 
        
        print(f'       +     |      -  ')       
        print(f'+ |  {tpos},  {fneg} ')
        print(f'- |  {fpos},  {tneg} ')
        sens = tpos / (tpos + fneg)
        spec = tneg / (tneg + fpos)
        acc = (100 * correct/total)

        fps, tps, threshold = roc_curve(targets, softpred[:])

        conf_matrix = confusion_matrix(prediction, targets)

    return (
        conf_matrix,        # Test Confusion Matrix for Malignant and Benign Nodules
        fps,                # False Positive Rates
        tps,                # True Positiive Rates
        sens,               # Sensitivity
        spec,               # Specificity
        acc,                # Accuracy of network
    )
