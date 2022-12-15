# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #
from argparse import Namespace
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
import torch
import sys, os

from ParenchymalAttention.networks import eval
from ParenchymalAttention.networks import architectures
from ParenchymalAttention.data import dataloader
from ParenchymalAttention.utils import image
from ParenchymalAttention.utils import utils
from ParenchymalAttention.utils import progress
from ParenchymalAttention.utils import metrics
# ---------------------------------------------------------------------------- #

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
        net = architectures.MobileNetV1()
        net.init_weights()
    
    elif (model == 'MobileNet2'):
        net = architectures.MobileNetV2()
        net.init_weights()
    
    elif (model == "Miniception"):
        net = architectures.Miniception()
        net.init_weights()

    else:
        print("Warning: Model Not Found")
    
    return net

def experiment(dataset:object, method:str, config:object, masksize:int):
    """
    Description:
    -----------
    Parameters:
    """
    # Initializing One-off variables
    best_acc = 0
    fig = 0
    
    # Defining empty lists to store network performance information
    trainloss, valloss =  [], []
    trainacc, valacc =  [], []
    sensitivity =  np.zeros((config['experiment']['folds'],config['experiment']['reps']))
    specificity =  np.zeros((config['experiment']['folds'],config['experiment']['reps']))
    auc_scores = np.zeros((config['experiment']['folds'],config['experiment']['reps']))
    train_time = np.zeros((config['experiment']['folds'],config['experiment']['reps']))
    

    if method != 'Otsu' or method != 'BlockDrop':
        bar =progress.ProgressBar(model= config['experiment']['model'], method= method,
                              maxfold= config['experiment']['folds'],
                              maxrep= config['experiment']['reps'],
                              maskratio= masksize,
                              bar_length= 50)
    else:
        bar =progress.ProgressBar(model= config['experiment']['model'], method= method,
                              maxfold= config['experiment']['folds'],
                              maxrep= config['experiment']['reps'],
                              maskratio=None,
                              bar_length= 50)


    for k in range(config['experiment']['folds']):
        df_train, df_test = train_test_split(dataset, test_size= config['experiment']['split'][2], random_state = k)
        df_train, df_val = train_test_split(df_train, test_size= config['experiment']['split'][1], random_state = k)
                
        fprs, tprs = [], []

        for r in range(config['experiment']['reps']):
            bar._update(rep= r, fold= k)

            #Initialize Net
            net = net_select(config['experiment']['model'])
            net = net.to(config['device'])
            
            if config['flags']['Params'] and k==0 and r==0:
                utils.check_parameters([net], params={'model': [config['experiment']['model']]})

            bar.info()

            # Load Training Dataset
            trainset = dataloader.DFLoader(df_train, method=method)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size= 125, shuffle= True)

            valset = dataloader.DFLoader(df_val, method=method)
            valloader = torch.utils.data.DataLoader(valset, batch_size= 125, shuffle= True)
            
            #Load testing Dataset
            testset = dataloader.DFLoader(df_test, method=method)
            testloader = torch.utils.data.DataLoader(testset, batch_size= 125, shuffle= True)

            output = eval.train(trainloader, valloader, net, bar, config)
            
            trainloss.append(output['Trainingloss'])
            valloss.append(output['Validationloss'])
            trainacc.append(output['TrainingAccuracy'])
            valacc.append(output['ValidationAccuracy'])

            confmatrix, fp, tp, sens, spec, acc, TNoimg, TPoimg, FNoimg, FPoimg = eval.test(testloader, net, device=config['device'], CreateComposites=config['flags']['CreateComposites'])   

            if acc > best_acc:
                best_acc = acc
                utils.model_save(method, masksize, net)
                metrics.plot_confusion_matrix(confmatrix, config['experiment']['classnames'], r, masksize, method, normalize = True, saveFlag = True)
                
                metrics.plot_metric(params={
                                    'xlabel': 'epochs',
                                    'ylabel': 'Loss',
                                    'title': 'Autoencoder Loss (Mask Ratio: %s)'%(str(masksize)),
                                    'maskratio': masksize,
                                    'trainmetric': output['Trainingloss'],
                                    'valmetric': output['Validationloss'],
                                    'legend': ['Training','Validation'],
                                    'savename': 'Network_Loss',
                                    'method': method,
                                    })

                metrics.plot_metric(params= {
                                    'xlabel': 'epochs',
                                    'ylabel': 'Accuracy',
                                    'title': 'Network Accuracy (Mask Ratio: %s)'%(str(masksize)),
                                    'maskratio': masksize,
                                    'trainmetric': output['TrainingAccuracy'],
                                    'valmetric': output['ValidationAccuracy'],
                                    'legend': ['Training','Validation'],
                                    'savename': 'Class_Accuracy',
                                    'method': method,
                                    })

                if config['flags']['CreateComposites']:                     
                    utils.saveAttentionImg(TNoimg, method, masksize, title= 'AverageTrueNegative_Image')
                    utils.saveAttentionImg(TPoimg, method, masksize, title= 'AverageTruePositive_Image')
                    utils.saveAttentionImg(FNoimg, method, masksize, title= 'AverageFalseNegative_Image')
                    utils.saveAttentionImg(FPoimg, method, masksize, title= 'AverageFalsePositive_Image')

                if config['flags']['Standards']:
                    selected_images = [
                                    '100463_0_x.jpg',  # Benign Nodule
                                    '100397_0_x.jpg',  # Benign Nodule + Parenchymal abnormalities
                                    '101606_0_x.jpg',  # Benign Nodule + Plural Wall
                                    '100012_1_x.jpg',  # Malignant Nodule
                                    '100570_1_y.jpg',  # Malignant Nodule + Parenchymal abnormalities
                                    '101692_1_y.jpg',  # Malignant Nodule + Plural Wall
                    ]
                    for img in selected_images:
                        imfile = os.path.split(os.getcwd())[0] + '/dataset/' + img
                        image.getcam(imfile, masksize, net, method, config['device'], fileid = img)


                image.getcam(testloader, masksize, net, method, config['device'], folder = '/GradCAM/')

            sensitivity[k,r] = sens
            specificity[k,r] = spec

            fprs.append(fp), tprs.append(tp)
        
        auc_scores[k,:] = metrics.calcAuc(fprs,tprs, method, masksize, r, plot_roc= True)
        fig += 1

    utils.csv_save(method, masksize, sensitivity, name = 'sensitivity')
    utils.csv_save(method, masksize, specificity, name = 'specificity')
    utils.csv_save(method, masksize, auc_scores, name = 'auc')             


def main(args, command_line_args):

    # Network Parameters
    config= {
        'device': GPU_init(0),
        'experiment':{
            'seed': args.seed,
            'data': args.data,
            'reps': args.reps,
            'folds': args.folds,
            'split': args.split,
            'model': args.model,
            'method': args.method,
            'masksize': args.masksize,
            'classnames': args.classes,
        },
        'opt':{
            'loss': args.loss,
            'optimizer': args.optim,
            'epochs': args.epochs,
            'lr': args.lr,
            'betas': args.betas,
            'rho': args.rho,
            'eps': args.eps,
            'decay': args.decay,
            'momentum': args.momentum
        },
        'flags':{
            'CreateComposites': args.composites,
            'EvalBestModel': args.bestmodel,
            'Standards': args.standards,
            'Features': args.features,
            'Concepts': args.concepts,
            'Params': args.params,
            'SaveFigs': args.savefigures,
            'SaveAUC': args.saveAUC,
        }
    
    }

    sys.stdout.write('\n\r {0}\n Loading User data from: {1}\n {0}\n '.format('='*(24 + len(config['experiment']['data'])), config['experiment']['data']))
    
    dataset = dataloader.load_files(config)

    for method in config['experiment']['method']:
        if method== 'Otsu' or method=='DropBlock':
            for masksize in config['experiment']['masksize']:
                experiment(dataset=dataset, method=method, config=config, masksize=masksize)
        else:
            experiment(dataset=dataset, method=method, config=config, masksize=None)
            



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Experiment Settings
    parser.add_argument('--data', type=str, required=True, help='Absolute path for data directory')
    parser.add_argument('--seed', type=int, default= 2022, help='Random Seed for Initializing Network')
    parser.add_argument('--reps', type=int, default=1, help='Number of repetition for a given fold')
    parser.add_argument('--folds', type=int, default=1, help='Number of Folds')
    parser.add_argument('--split', type=tuple, default=(0.7,0.15,0.15), help='Dataset training/validation/testing split')
    parser.add_argument('--model', type=str, default='Miniception', choices=['Miniception','MobileNetV1','MobileNetV2'])
    parser.add_argument('--method', type=list, default=['Original','Tumor-Segmentation','Surround-Segmentation'], 
                        choices=['Original','Tumor-Segmentation','Surround-Segmentation','Otsu','DropBlock'])
    parser.add_argument('--masksize', type=list, default=[16,32,48,64], help='Size of area that will be blocked using Otsu or DropBlock')
    parser.add_argument('--classes', type=list, default=['Benign','Malignant'])

    # Optimizer Arguments
    parser.add_argument('--loss', type=str, default='entropy')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--betas',type=tuple, default=(0.9,0.999))
    parser.add_argument('--rho',type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=1e-10)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.99)
    # Experiment Flags
    parser.add_argument('--composites', type=bool, default=True, help='Create Composite Images of false negatives, false positives, etc...')
    parser.add_argument('--bestmodel', type=bool, default=True, help='Evaluate Best Model')
    parser.add_argument('--standards', type=bool, default=False, help='Evaluate Cherry Picked Images')
    parser.add_argument('--features', type=bool, default=False, help='Not Implemented')
    parser.add_argument('--concepts', type=bool, default=False, help='Not Implemented')
    parser.add_argument('--params', type=bool, default=True, help='Print number of parameters in Network')
    parser.add_argument('--savefigures', type=bool, default=True, help='Save Figures generated (Loss over epoch, etc...)')
    parser.add_argument('--saveAUC', type=bool, default=True, help='Save AUC Figures and data')

    return parser
     
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()
    main(args, command_line_args=sys.argv)
    