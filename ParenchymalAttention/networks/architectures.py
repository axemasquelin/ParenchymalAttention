# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #

from collections import OrderedDict
from torch import Tensor
from re import X

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
import ParenchymalAttention.utils.utils as utils
# ---------------------------------------------------------------------------- #

class BasicConv2d(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, nxn: int, pad: int=1, stride: int=1, dilate: int=1):
        super().__init__()

        self.Conv = nn.Conv2d(channel_in, channel_out, nxn, padding = pad, stride= stride, dilation = dilate, bias = False)
        self.BatchNorm = nn.BatchNorm2d(channel_out, eps = 0.0001)
        self.ReLu = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding=1)

    @torch.jit.export    
    def forward(self, x: Tensor, relu: bool = False) -> Tensor:
        x = self.Conv(x)
        x = self.BatchNorm(x)
        x = self.ReLu(x)
        return self.MaxPool(x)

class NaiveInception(nn.Module):
    def __init__(self, chn1x_in: int, chn1x_out: int,
                chn3x_in: int, chn3x_out: int,
                chn5x_in: int, chn5x_out: int):
        super().__init__()

        self.branch1x1 = BasicConv2d(channel_in= chn1x_in, channel_out= chn1x_out, nxn= 1, pad = 0, dilate= 1) 
        self.branch3x3 = BasicConv2d(channel_in= chn3x_in, channel_out= chn3x_out, nxn= 3, pad = 1, dilate= 1) 
        self.branch5x5 = BasicConv2d(channel_in= chn5x_in, channel_out= chn5x_out, nxn= 5, pad = 2, dilate= 1)
        self.branchpool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding=1)
    
    @torch.jit.export
    def _forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branchpool = self.branchpool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branchpool]
        return outputs
    
    @torch.jit.export
    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        return outputs

class Naiveception(nn.Module):
    def __init__(self, x_in:int=1, conv1out:int=16, n_classes:int = 2, avgpool_kernel:int = 4):
        super(Naiveception, self).__init__()
        def simple_convs(x_in:int=1, x_mid:int=3, x_out:int=3, nxn:int=7, stride:int=2):
            return nn.Sequential(
                nn.Conv2d(x_in, x_mid, kernel_size=nxn, stride=stride, padding=2, bias=False),
                nn.BatchNorm2d(x_mid, eps = 0.0001),
                nn.ReLU(inplace = True),
                nn.Conv2d(x_mid, x_out, kernel_size=nxn-2, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(x_out, eps = 0.0001),
                nn.ReLU(inplace = True),
                # nn.Conv2d(x_mid*2, x_out, kernel_size=nxn-4, stride=stride, padding=0, bias=False),
                # nn.BatchNorm2d(x_out, eps = 0.0001),
                # nn.ReLU(inplace = True),
            )
        
        self.avgpool_kernel = avgpool_kernel
        self.conv1 = simple_convs(x_in=1, x_mid=3, x_out=conv1out,nxn=5, stride=2)
              
        self.config=[
            # 1ach_in, 1ach_out, 3ch_in, 3ch_out, 5ch_in, 5ch_out
            [conv1out, 67, conv1out, 67, conv1out, 67],
            # [56, 67 , 56, 67, 56, 67],
            [217, 85, 217, 85, 217, 85],
        ]

        layers = []

        for c1x_in, c1x_out, c3x_in, c3x_out, c5x_in, c5x_out in self.config:
            layers.append(NaiveInception(chn1x_in=c1x_in, chn1x_out=c1x_out,
                                           chn3x_in=c3x_in, chn3x_out=c3x_out,
                                           chn5x_in=c5x_in, chn5x_out=c5x_out
                                           )
                        )

        self.dims = self.config[-1][0]+ self.config[-1][1]*3
        
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(self.avgpool_kernel)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.dims*self.avgpool_kernel**2, 1028),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(1028, 128),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
            nn.Sigmoid()
        )
    
    @torch.jit.ignore
    def init_weights(m):
        '''Initializes Model Weights using Xavier Uniform Function'''
        np.random.seed(2020)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    @torch.jit.ignore
    def adjust_lr(optimizer, lrs, epoch):
        lr = lrs * (0.01 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def forward(self, x, flag = False):
        x = self.conv1(x)
        x = self.layers(x)
        x_avg = self.avg_pool(x)
        x = x_avg.view(-1, self.dims*self.avgpool_kernel**2)
        x = self.classifier(x)

        if flag:
            return x, x_avg
        
        return x 

class conv_depth(nn.Module):
    def __init__(self, x_in:int, x_out:int, stride:int, dilation:int):
        super().__init__()
                        
        self.conv_depthwise = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels= x_in, out_channels=x_in, kernel_size=3,
                          stride=stride, padding=1, groups=x_in, dilation=dilation, bias = False),
                nn.BatchNorm2d(x_in),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 1, padding=1),


                # Pointwise Convolution
                nn.Conv2d(x_in, x_out, (1,1), stride = 1, bias = False),
                nn.BatchNorm2d(x_out),
                nn.ReLU(inplace = True)
                )
    def forward(self, x:Tensor):
        return self.conv_depthwise(x)
    
class MobileNetV1(nn.Module):
    def __init__(self, chn_in:int=1, n_class:int=2, avg_kernel:int=2):
        '''
        Architecture following the MobileNet version 1 schema to allow depthwise convolution. The architecture is slightly modified to 
        allow or remove the pixelwise convolution that occures in the average pool layer.
        -----------
        Parameters:
        chn_in - integer
            chn_in describes the number of channels present in the input image in order to initialize network to correct input dimensions
        chn_out - integer
            chn_out refers to the number of classes the network will be prediciting. The model is original trained to have two classes (Benign and Malignant) classification
        avg_kernel - integer
            avg_kernel controls whether a pixelwise convolution occurs at the end of the network or a larger convolution is applied. 
        '''
        super(MobileNetV1, self).__init__()
        self.avg_kernel = avg_kernel
        def conv_full(x, x_out, stride):
            return nn.Sequential(
                nn.Conv2d(x, x_out, 3, stride, 1, bias = False),
                nn.BatchNorm2d(x_out),
                nn.ReLU(inplace = True),
            )

        self.Netconfig=[
           #in, out, stride, dilation
            [8, 16, 1, 1],
            [16, 32, 2, 3],
            [32, 32, 1, 1],
            [32, 64, 2, 2],
            [64, 128, 2, 1],
            # [128, 256, 2],
            # [256, 256, 1],
        ]

        self.full_conv = conv_full(chn_in, self.Netconfig[0][0], 2)

        layers = []

        for  x_in, x_out, stride, dilation in self.Netconfig:
            layers.append(conv_depth(x_in=x_in, x_out=x_out, stride=stride, dilation=dilation))
        
        self.max_size = self.Netconfig[-1][0]
        self.layers = nn.Sequential(*layers)
        self.lastconv = conv_full(self.Netconfig[-1][1],self.max_size, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(avg_kernel)

        self.fullyconnected = nn.Sequential(
            nn.Dropout(.65),
            nn.Linear(self.max_size*self.avg_kernel**2, self.max_size*self.avg_kernel),
            nn.LeakyReLU(inplace=True),
            
            nn.Dropout(.65),
            nn.Linear(self.max_size*self.avg_kernel, self.max_size),
            nn.LeakyReLU(inplace=True),

            nn.Dropout(.5),
            nn.Linear(self.max_size, n_class),
            nn.Sigmoid()
            )

    def init_weights(m):
        '''Initializes Model Weights using Xavier Uniform Function'''
        np.random.seed(2020)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x, flag=False):
        x = self.full_conv(x)
        x = self.layers(x)
        x = self.lastconv(x)
        x_avg = self.avgpool(x)
        x = x_avg.view(-1, self.max_size*self.avg_kernel**2)
        x = self.fullyconnected(x)
        
        if flag:
            return x, x_avg
        return x
