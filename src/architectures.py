# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #
from torch import Tensor
from audioop import bias
from re import X
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

import torch.optim as optim
import torch.nn as nn
import torchvision
import torch

import numpy as np
import utils
# ---------------------------------------------------------------------------- #
def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def adjust_lr(optimizer, lrs, epoch):
    lr = lrs * (0.01 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
                
        def conv_full(x, x_out, stride):
            return nn.Sequential(
                nn.Conv2d(x, x_out, 3, stride, 1, bias = False),
                nn.BatchNorm2d(x_out),
                nn.ReLU(inplace = True)
            )
        
        def conv_depthwise(x, x_out, stride): 
            return nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(x, x, (3,3) , stride, 1, groups = x, bias = False),
                nn.BatchNorm2d(x),
                nn.ReLU(inplace = True),
                

                # Pointwise Convolution
                nn.Conv2d(x, x_out, (1,1), stride = 1, bias = False),
                nn.BatchNorm2d(x_out),
                nn.ReLU(inplace = True)
                )
        
        self.model = nn.Sequential(
            conv_full(1, 32, 2),
            conv_depthwise(32, 64, 1),
            conv_depthwise(64, 128, 2),
            conv_depthwise(128, 128, 1),
            conv_depthwise(128, 256, 2),
            conv_depthwise(256,256, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fullyconnected =nn.Linear(256, 2)

    def init_weights(m):
        '''Initializes Model Weights using Xavier Uniform Function'''
        np.random.seed(2020)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def adjust_lr(optimizer, lrs, epoch):
        lr = lrs * (0.01 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        m.bias.data.fill_(0.01)

    def forward(self, x, flag = False):
        x_avg = self.model(x)
        x = x_avg.view(-1, 256)
        x = self.fullyconnected(x)
        
        if flag:
            return x, x_avg
        
        else:
            return x

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])
        nn.ReLU(inplace = True),
        
class MobileNetV2(nn.Module):
    def __init__(self, ch_in=3, n_classes=1000):
        super(MobileNetV2, self).__init__()

        self.configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.stem_conv = conv3x3(ch_in, 32, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1(input_channel, 1280)

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(1280, n_classes)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 2)
        x = self.classifier(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, chn_in: int, chn_out: int, nxn: int, pad: int, dilate: int):
        super().__init__()

        self.Conv = nn.Conv2d(chn_in, chn_out, nxn, padding = pad, dilation = dilate, bias = False)
        self.BatchNorm = nn.BatchNorm2d(chn_out, eps = 0.001)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.BatchNorm(x)

        return F.relu(x, inplace=True)

class NaiveInception(nn.Module):
    def __init__(self, chn1x_in: int, chn1x_out: int,
                chn3x_in: int, chn3x_out: int,
                chn5x_in: int, chn5x_out: int):
        super().__init__()

        self.branch1x1 = BasicConv2d(chn_in= chn1x_in, chn_out= chn1x_out, nxn= 1, pad = 0, dilate= 1) 
        self.branch3x3 = BasicConv2d(chn_in= chn3x_in, chn_out= chn3x_out, nxn= 3, pad = 1, dilate= 1) 
        self.branch5x5 = BasicConv2d(chn_in= chn5x_in, chn_out= chn5x_out, nxn= 5, pad = 2, dilate= 1)
        self.branchpool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding=1)

    def _forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branchpool = self.branchpool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branchpool]

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class DRInceptionblock(nn.Module):
    def __init__(self, chn1x_in: int, chn1x_out: int,
                chn3x_in: int, chn3x_out: int,
                chn5x_in: int, chn5x_out: int) -> None:
        super().__init__()

        self.branch1x1a = BasicConv2d(chn_in= chn1x_in, chn_out= chn1x_out, nxn= 1, pad = 0)
        self.branch3x3 = BasicConv2d(chn_in= chn3x_in, chn_out= chn3x_out, nxn= 3, pad = 1) 
        self.branch5x5 = BasicConv2d(chn_in= chn5x_in, chn_out= chn5x_out, nxn= 5, pad = 2)
        self.branch3x3pool = nn.MaxPool2d((3,3))

    def _forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1a(x)
        branch3x3 = self.branch3x3(self.branch1x1a(x))
        branch5x5 = self.branch5x5(self.branch1x1a(x))
        branchpool = self.branch1x1a(self.branch3x3pool(x))

        outputs = [branch1x1, branch3x3, branch5x5, branchpool]

        return outputs
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs,1)


class Miniception(nn.Module):
    def __init__(self, n_classes = 2):
        super(Miniception, self).__init__()
        
        self.Naiveconfig=[
            # 1ach_in, 1ach_out, 3ch_in, 3ch_out, 5ch_in, 5ch_out
            [1, 8, 1, 8, 1, 8],
            [25, 12, 25, 12, 25, 12]
        ]

        # self.DRconfig=[
        #     # 1ach_in, 1ach_out, 1b_in, 1b_out, 3ch_in, 3ch_out, 5ch_in, 5ch_out
        #     [1, 4, 4, 4, 4, 4],
        #     [16, 8, 8, 12, 8, 12]
        # ]

        layers = []
        for c1x_in, c1x_out, c3x_in, c3x_out, c5x_in, c5x_out in self.Naiveconfig:
            layers.append(NaiveInception(chn1x_in=c1x_in, chn1x_out=c1x_out,
                                           chn3x_in=c3x_in, chn3x_out=c3x_out,
                                           chn5x_in=c5x_in, chn5x_out=c5x_out
                                           )
                        )
        
        self.layers = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool2d((32,32))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(49*32*32, 1000),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(1000, 275),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(275, 49),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(49, n_classes)
        )

    def init_weights(m):
        '''Initializes Model Weights using Xavier Uniform Function'''
        np.random.seed(2020)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def adjust_lr(optimizer, lrs, epoch):
        lr = lrs * (0.01 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def forward(self, x, flag = False):
        x = self.layers(x)
        x_avg = self.avg_pool(x)
        x = x_avg.view(-1, 49*32*32)
        x = self.classifier(x)

        if flag:
            return x, x_avg
        
        return x 

