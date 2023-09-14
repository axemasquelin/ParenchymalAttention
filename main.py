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
from torchinfo import summary

import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
import torch
import sys, os

from ParenchymalAttention.networks import eval
from ParenchymalAttention.networks import architectures
from ParenchymalAttention.data import dataloader
from ParenchymalAttention.utils import utils
from ParenchymalAttention.utils import progress
from ParenchymalAttention.utils import metrics
from ParenchymalAttention import postprocess
# ---------------------------------------------------------------------------- #

def GPU_init(loc):
    """
    Definition: GPU Initialization function
    -----------
    Parameters:
    loc - int
        0 or 1 depending on which gpu is being utilized. In the case of a bridge, it is possible to further parallerize the network
    --------
    Returns:
    check_gpu - int
        gpu enable variable that will be utilized as the device/location we are sending data to.
    """
    check_gpu = torch.device("cuda:" + str(loc) if torch.cuda.is_available() else "cpu")    # torch.device defines whether a gpu is used or cpu depending on availability of gpus
    print("Available Device: " + str(check_gpu))
    
    return check_gpu

def net_select(model):
    """
    Network selection function customized to allow initialization from scratch of custom network architectures.
    -----------
    Parameters:
    model - string
        defines which model will be used based on predefined architectures names and if statement
    --------
    Returns:
    net - class 
        class containing the parameters and information of the network. Check architecture.py for further information on custom networks
    """
    if (model == 'EfficientNet'):
        net = architectures.EfficientNet()
        net.apply(net.init_weights)
    
    elif (model == "Miniception"):
        net = architectures.Naiveception()
        net.init_weights()

    elif (model == 'MobileNet'):
        net = architectures.MobileNetV1()
        net.init_weights()
    else:
        print("Warning: Model Not Found")
    
    return net

def result_directories(method:str):
    """
    Creates the expected directories within results in case they are missing
    -----------
    Parameters:
    method - string
        variable containing string of the method being evaluated at the moment
    """
    utils.create_directories(folder = '/results/' + method)
    utils.create_directories(folder = '/results/' + method + '/GradCAM')
    utils.create_directories(folder = '/results/' + method + '/Networks')


def experiment(dataset:object, method:str, config:object, masksize:int):
    """
    Function controlling all experiment component of defined by the parsed function call. The experiment function does not return anything to the main function.
    At the place, it will save all results for the given methodology to its respective result folder(s). See github repo to ensure proper directories exists.
    TODO: automatically create expected results directories if not present. 
    -----------
    Parameters:
    dataset - pd.dataframe
        contains a dataframe with the filename(s), classification(s), and other metrics of interest by the user. 
        The dataframe is created when calling dataframe.load_files() and can be viewed in 
        /ParenchymalAttention/data/dataloader.py
    method - string
        containes experiment information regarding which method is currently being evaluated and applied to the input images.
        In the case of these experiment, the options include Original image, Otsu algo, DropBloc algo, and Segmentation Masks. 
    config - set
        contains the experiment configuration and network hyperparameters that user defines in function call. See build_parser() function
        or main()
    masksize - int
        integer defining how much of the image will be removed when applying the Otsu algorithm, or DropBlock algorithm.
    """
    # Initializing One-off variables
    best_acc = 0
    fig = 0
    
    # Defining empty lists to store network performance information
    trainloss, valloss =  [], []
    trainacc, valacc =  [], []
    fprs, tprs = [], []
    dataset_split = []

    sensitivity =  np.zeros((config['experiment']['folds'],config['experiment']['reps']))
    specificity =  np.zeros((config['experiment']['folds'],config['experiment']['reps']))
    auc_scores = np.zeros((config['experiment']['folds'],config['experiment']['reps']))
    

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
        df_train, df_test = train_test_split(dataset, test_size= config['experiment']['split'][1], random_state = k)
        # df_train, df_val = train_test_split(df_train, test_size= config['experiment']['split'][1], random_state = k)
        
        df_train = dataloader.augment_dataframe(df_train, upsample=3, augment='rand')
        # df_val = dataloader.augment_dataframe(df_val, upsample=6, augment='rand')
        df_test = dataloader.augment_dataframe(df_test, upsample=2, augment='infer')


        for r in range(config['experiment']['reps']):
            bar._update(rep= r, fold= k)

            #Initialize Net
            net = net_select(config['experiment']['model'])
            net = net.to(config['device'])
            
            if config['flags']['Params'] and k==0 and r==0:
                summary(net, (1,1,64,64))

            # Load Training Dataset
            trainset = dataloader.NPYLoader(df_train, method=method)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size= 128, shuffle= True)
            
            # valset = dataloader.NPYLoader(df_val, method=method)
            # valloader = torch.utils.data.DataLoader(valset, batch_size= 125, shuffle= True)

            #Load testing Dataset
            testset = dataloader.NPYLoader(df_test, method=method, testing=True)
            testloader = torch.utils.data.DataLoader(testset, batch_size= 128, shuffle= True)

            output = eval.train(trainloader, testloader, net, bar, config)
            print(output['TrainingAccuracy'])
            print(output['Trainingloss'])
            trainloss.append(output['Trainingloss'])
            valloss.append(output['Validationloss'])
            trainacc.append(output['TrainingAccuracy'])
            valacc.append(output['ValidationAccuracy'])

            confmatrix, fp, tp, sens, spec, acc = eval.test(testloader, net, device=config['device'])   

            """
            image.getcam(testloader, net, method,
                        fold = k, rep = r,
                        masksize= masksize,
                        selectcam='GradCAM',
                        device=config['device'],
                        folder='/GradCAM/')       
            """            
            
            utils.model_save(method, masksize, fold=k, rep=r, net=net)

            if acc > best_acc:
                best_acc = acc
   
                metrics.plot_confusion_matrix(confmatrix, config['experiment']['classnames'], r, masksize, method, normalize = True, saveFlag = True)
                
                metrics.plot_metric(params={
                                    'xlabel': 'epochs',
                                    'ylabel': 'Loss',
                                    'title': '%s Loss (Mask Ratio: %s)'%(str(config['opt']['loss']),str(masksize)),
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

            sensitivity[k,r] = sens
            specificity[k,r] = spec
            
            dataset_split.append(f"Training: {len(df_train)} | Test: {len(df_test)}")
            fprs.append(fp), tprs.append(tp)
        
    auc_scores = np.reshape(metrics.calcAuc(fprs,tprs, method, masksize, r, plot_roc= True),
                            (config['experiment']['folds'],config['experiment']['reps']))
    
    dataset_split = np.reshape(np.asarray(dataset_split),(config['experiment']['folds'],config['experiment']['reps']))

    utils.csv_save(method, masksize, dataset_split, name = 'dataset_split')
    utils.csv_save(method, masksize, sensitivity, name = 'sensitivity')
    utils.csv_save(method, masksize, specificity, name = 'specificity')
    utils.csv_save(method, masksize, auc_scores, name = 'auc')


    


def main(args, command_line_args):
    """
    Main.py controls all experiments for the Parenchymal Attention research. In this project, we aim to gain
    inisghts at how CNNs explore low dose computed tomography (LDCT) images in order to identify malignant and
    benign nodules between 4mm-20mm in diameter. This code can be repurposed to any tasks by simply changing
    the provided input data
    -----------
    Parameters:
    args - dictionary
        ***containing input arguments either from command_line inputs or TODO from provided yaml experiment file
    
    """
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
    
    dataset = dataloader.load_files(config, ext='.tif')

    for method in config['experiment']['method']:
        if method== 'Otsu' or method=='DropBlock':
            for masksize in config['experiment']['masksize']:
                experiment(dataset=dataset, method=method, config=config, masksize=masksize)
        else:
            result_directories(method)
            experiment(dataset=dataset, method=method, config=config, masksize=None)
    
    postprocess.run_model(config, dataset)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Experiment Settings
    parser.add_argument('--data', type=str, required=True, help='Absolute path for data directory')
    parser.add_argument('--seed', type=int, default= 2020, help='Random Seed for Initializing Network')
    parser.add_argument('--reps', type=int, default=10, help='Number of repetition for a given fold')
    parser.add_argument('--folds', type=int, default=10, help='Number of Folds')
    parser.add_argument('--split', type=tuple, default=(0.75,0.25), help='Dataset training/validation/testing split')
    parser.add_argument('--model', type=str, default='Miniception', choices=['Miniception','EfficientNet','MobileNet'])
    parser.add_argument('--method', type=list, default=['Original','Tumor-only','Parenchyma-only'])

    parser.add_argument('--masksize', type=list, default=[16,32,48,64], help='Size of area that will be blocked using Otsu or DropBlock')
    parser.add_argument('--classes', type=list, default=['Benign','Malignant'])

    # Optimizer Arguments
    parser.add_argument('--loss', type=str, default='entropy')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--betas',type=tuple, default=(0.9,0.999))
    parser.add_argument('--rho',type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=1e-7)
    parser.add_argument('--decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.99)
    # Experiment Flags
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
    