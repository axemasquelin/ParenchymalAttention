# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description: Image manipulation and visualization function
'''
# Libraries
# ---------------------------------------------------------------------------- #
# from PIL import Image
from sklearn.model_selection import train_test_split
from ParenchymalAttention.networks import architectures
from ParenchymalAttention.data import dataloader
from ParenchymalAttention.utils import image
from ParenchymalAttention.utils import utils

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import logging
import torch
import os, csv, glob
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

def RMSE_values(row:pd.DataFrame, mean_original, mean_surround, mean_tumor):
    """
    Calculates the Root Mean Square Error of the iteration
    """
    rmse = (row['Original'] - mean_original)**2 + (row['Parenchyma'] - mean_surround)**2 + (row['Tumor'] - mean_tumor)**2

    return rmse

def get_idx(i):
    """
    Finds the fold and repetition value for the lowest root mean square error
    """

    fold = i % 10
    repetition = i // 10

    return repetition, fold

def load_csv():

    models = [
            'Original',
            'Tumor-only',
            'Parenchyma-only'
            ]
    metric = 'auc'

    df = pd.DataFrame()                # General Dataframe to generate Bar-graph data
    np_obf = np.zeros((5,5))           # Conv Layer Dataframe for violin plots
    np_orig = np.zeros((5,5))          # Wavelet Layer Dataframe for violin plots
    np_comb = np.zeros((5,5))          # Multi-level Wavelet Dataframe
    
    for model in models:
        for root, dirs, files in os.walk(os.getcwd() + "/results/" + model + '/', topdown = True):
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
                        if metric == 'auc':
                            if (header == 'Original'):
                                np_orig = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                            if (header == 'Parenchyma-only'):
                                np_surround = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                            if (header == 'Tumor-only'):
                                np_tumor = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
    df = df.rename({'Parenchyma-only': 'Parenchyma', 'Tumor-only': 'Tumor'}, axis = 'columns')      
    cols = df.columns.tolist() 
    
    df = df[cols]
    return df

def find_representative(iterations:int=100):
    """
    Find representative identified the fold and repatitition to be utilized that minimizes the Root mean square error of the system by looking
    at the average of all approaches and finding a fold that minimize the loss across all methodologies
    -----------
    Parameters:
    mean_df - pd.DataFrame() #TODO: Feeling Cute Might Delete Later

    iterations -int
        Number of iterations run (Number of Folds * Number of Repetitions)
    --------
    Returns:
    iteration - int
    fold - int
    """
    df = load_csv()
    mean_original = df['Original'].mean()
    mean_surround = df['Parenchyma'].mean()
    mean_tumor = df['Tumor'].mean()
    
    rmse_i = [RMSE_values(df.iloc[x], mean_original, mean_surround, mean_tumor) for x in range(iterations)]
    rmse_i = np.asarray(rmse_i)
    rep,fold = get_idx(rmse_i.argmin())
    print("Representatitve", rep, fold)

    return rep, fold

def load_model(config, method, rep, fold):
    """
    """
    modelpath = os.getcwd() + '/results/' + method + f'/Networks/{str(fold)}_{str(rep)}/'

    net = architectures.Naiveception()

    network_names = glob.glob(modelpath + '*.pt')
    checkpoint = torch.load(network_names[0])

    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.to(config['device'])

    return net


def run_model(config, dataset):
    """
    """
    rep, fold = find_representative(iterations=config['experiment']['folds']*config['experiment']['reps'])

    df_train, df_test = train_test_split(dataset, test_size= config['experiment']['split'][1], random_state = fold)
    # df_train, df_val = train_test_split(df_train, test_size= config['experiment']['split'][1], random_state = k)
    df_test = dataloader.augment_dataframe(df_test, upsample=1, augment='infer')


    for method in config['experiment']['method']:
        utils.create_directories(folder= f'/results/{method}/GradCAM/{str(fold)}_{str(rep)}')
        testset = dataloader.NPYLoader(df_test, method=method, testing=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size= 128)
        net = load_model(config, method,rep,fold)

        image.getcam(testloader, net, method,
                     fold, rep,
                    masksize= None,
                    selectcam='GradCAM',
                    device=config['device'],
                    folder='/GradCAM/')       


def build_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser()

    # Experiment Settings
    parser.add_argument('--data', type=str, required=True, help='Absolute path for data directory')
    parser.add_argument('--seed', type=int, default= 2020, help='Random Seed for Initializing Network')
    parser.add_argument('--split', type=tuple, default=(0.75,0.25), help='Dataset training/validation/testing split')
    parser.add_argument('--model', type=str, default='Miniception', choices=['Miniception','EfficientNet','MobileNet'])
    parser.add_argument('--method', type=list, default=['Original','Tumor-only','Parenchyma-only'])
    parser.add_argument('--reps', type=int, default=10, help='Number of repetition for a given fold')
    parser.add_argument('--folds', type=int, default=10, help='Number of Folds')

    return parser

def build_config(args):
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
        }
    }
    return config

if __name__ == '__main__':
    """
    """
    parser = build_parser()
    config = build_config(parser.parse_args())
    dataset = dataloader.load_files(config, ext='.tif')
    print(len(dataset))

    run_model(config,dataset)
