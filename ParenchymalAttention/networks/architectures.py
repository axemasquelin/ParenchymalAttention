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

    def forward(self, x: Tensor, relu: bool = False) -> Tensor:

        x = self.Conv(x)
        x = self.BatchNorm(x)
        return self.ReLu(x)


class NaiveInception(nn.Module):
    def __init__(self, chn1x_in: int, chn1x_out: int,
                chn3x_in: int, chn3x_out: int,
                chn5x_in: int, chn5x_out: int):
        super().__init__()

        self.branch1x1 = BasicConv2d(channel_in= chn1x_in, channel_out= chn1x_out, nxn= 1, pad = 0, dilate= 1) 
        self.branch3x3 = BasicConv2d(channel_in= chn3x_in, channel_out= chn3x_out, nxn= 3, pad = 2, dilate= 2) 
        self.branch5x5 = BasicConv2d(channel_in= chn5x_in, channel_out= chn5x_out, nxn= 5, pad = 6, dilate= 3)
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


class DRInception(nn.Module):
    def __init__(self, chn1xa_in: int, chn1xa_out: int, 
                       chn3xa_out: int, chn3xb_out: int,
                       chn1xb_out: int, chnblock_out: int) -> None:
        super().__init__()
        chn1xb_in = chn1xa_out + chn3xa_out + chn3xb_out
        self.block1x1a = BasicConv2d(channel_in=chn1xa_in, channel_out= chn1xa_out, nxn= 1, pad= 0, dilate= 1)
        self.block3x3a = BasicConv2d(channel_in=chn1xa_out, channel_out= chn3xa_out, nxn= 3, pad= 1, dilate= 1)
        self.block3x3b = BasicConv2d(channel_in=chn3xa_out, channel_out= chn3xb_out, nxn= 3, pad= 1, dilate= 1)

        self.block1x1b = BasicConv2d(channel_in= chn1xb_in, channel_out= chn1xb_out, nxn= 1, pad= 0, dilate= 1)

        self.convblock = BasicConv2d(channel_in= chn1xb_out + chn1xa_in, channel_out= chnblock_out, nxn=3, pad=0, stride= 2, dilate= 2)
        self.ReLu = nn.ReLU()


    def _forward(self, x: Tensor) -> Tensor:
        branch1x = self.block1x1a(x)
        branch1x3 = self.block3x3a(self.block1x1a(x))
        branch1x3x3 = self.block3x3b(self.block3x3a(self.block1x1a(x)))
        
        # print(f'\n branch1x: {branch1x.shape}\
        #         branch1x3: {branch1x3.shape}\
        #         branch1x3x3: {branch1x3x3.shape}')

        combined = [branch1x, branch1x3, branch1x3x3]
        
        x = torch.cat(combined, 1)

        return self.block1x1b(x)
    
    def forward(self, x: Tensor) -> Tensor:
        output = self._forward(x)
        x = self.ReLu(torch.cat([x,output],1))
        x = self.convblock(x)
        return x


class Miniception(nn.Module):
    def __init__(self, n_classes:int = 2, avgpool_kernel:int = 16):
        super(Miniception, self).__init__()
        
        self.Naiveconfig=[
            # 1ach_in, 1ach_out, 3ch_in, 3ch_out, 5ch_in, 5ch_out
            [1, 4, 1, 4, 1, 4],
            [13, 4, 13, 4, 13, 4],
            [25, 4, 25, 4, 25, 4],
            [37, 4, 37, 4, 37, 4]
        ]

        self.DRconfig=[
            # Chn1xa_in, chn1xa_out, chn3xa_out, chn3xb_out, chn1xb_in, chn1xb_out, chnblock_out
            [1, 3, 5, 7, 15, 16],
            [16, 12, 24, 36, 50, 66],
            [66, 12, 24, 36, 50, 116],
            [116, 12, 24, 36, 50, 125],
        ]

        layers = []
        '''
        # for c1x_in, c1x_out, c3x_in, c3x_out, c5x_in, c5x_out in self.Naiveconfig:
        #     layers.append(NaiveInception(chn1x_in=c1x_in, chn1x_out=c1x_out,
        #                                    chn3x_in=c3x_in, chn3x_out=c3x_out,
        #                                    chn5x_in=c5x_in, chn5x_out=c5x_out
        #                                    )
        #                 )
        '''
        
        for  chn1xa_in, chn1xa_out, chn3xa_out, chn3xb_out, chn1xb_out, chnblock_out in self.DRconfig:
            layers.append(DRInception(chn1xa_in= chn1xa_in, chn1xa_out= chn1xa_out, 
                                      chn3xa_out= chn3xa_out, chn3xb_out= chn3xb_out,
                                      chn1xb_out= chn1xb_out, chnblock_out= chnblock_out)
                                      )
        
        self.dims = self.DRconfig[-1][-1]
        
        self.layers = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool2d((avgpool_kernel,avgpool_kernel))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.dims*16**2, 500),
            nn.ReLU(inplace = True),
            nn.Dropout(0.6),
            nn.Linear(500, self.dims),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(self.dims, n_classes),
            nn.Sigmoid()
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
        x = x_avg.view(-1, self.dims*16**2)
        x = self.classifier(x)

        if flag:
            return x, x_avg
        
        return x 