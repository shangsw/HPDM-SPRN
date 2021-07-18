# -*- coding: utf-8 -*-
import torch
import torch.nn as nn 
import torch.nn.functional as F
from utils import HPDM

# Original implementation of DFFN is based on Caffe. Code is available at https://github.com/weiweisong415/Demo_DFFN
# This is a PyTorch version of DFFN

class basic_block(nn.Module):
    #basic residual block
    def __init__(self, filters):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, (3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, (3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, input):
        x1 = F.relu(self.bn1(self.conv1(input)))
        x2 = self.bn2(self.conv2(x1))
        eltwise = F.relu(input + x2)

        return eltwise


class trans_block(nn.Module):
    #transition block between different stage
    def __init__(self, input_channels, filters):
        super(trans_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, filters, (1,1), padding=0, stride=1) #stride=2
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(input_channels, filters, (3,3), padding=1, stride=1) #stride=2
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv2_1 = nn.Conv2d(filters, filters, (3,3), padding=1, stride=1)
        self.bn2_1 = nn.BatchNorm2d(filters)

    def forward(self, input):
        x1 = self.bn1(self.conv1(input))
        x2 = F.relu(self.bn2(self.conv2(input)))
        x2_1 = self.bn2_1(self.conv2_1(x2))
        eltwise = F.relu(x1 + x2_1)

        return eltwise


class dffn(nn.Module):
    def __init__(self, bands, classes, layers_num, filters=16):
        super(dffn, self).__init__()
        self.conv1 = nn.Conv2d(bands, filters, (3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.stage1 = self._make_stage(layers_num[0], filters)

        self.trans1 = trans_block(filters, filters*2)
        self.stage2 = self._make_stage(layers_num[1], filters*2)
        
        self.trans2 = trans_block(filters*2, filters*4)
        self.stage3 = self._make_stage(layers_num[2], filters*4)

        self.conv_stage1 = nn.Conv2d(filters, filters*4, (3,3), padding=1, stride=1)    #stride=4
        self.bn_stage1 = nn.BatchNorm2d(filters*4)
        self.conv_stage2 = nn.Conv2d(filters*2, filters*4, (3,3), padding=1, stride=1)  #stride=2
        self.bn_stage2 = nn.BatchNorm2d(filters*4)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(filters*4, classes)

    def _make_stage(self, block_num, filters):
        layers = []
        for i in range(block_num):
            layers.append(basic_block(filters))

        return nn.Sequential(*layers)

    def forward(self, input):
        input = input.squeeze(1)
        #stage 1
        x = F.relu(self.bn1(self.conv1(input)))
        eltwise4 = self.stage1(x)

        #stage 2
        eltwise5 = self.trans1(eltwise4)
        eltwise8 = self.stage2(eltwise5)

        #stage 3
        eltwise9 = self.trans2(eltwise8)
        eltwise12 = self.stage3(eltwise9)
        
        #fusion
        conv_eltwise4 = self.bn_stage1(self.conv_stage1(eltwise4))
        conv_eltwise8 = self.bn_stage2(self.conv_stage2(eltwise8))

        fuse2 = conv_eltwise4 + conv_eltwise8 + eltwise12
        pool = self.avgpool(fuse2)
        out = self.fc(pool.flatten(1))

        return out


class hpdm_dffn(nn.Module):
    def __init__(self, bands, classes, layers_num):
        super(hpdm_dffn, self).__init__()
        self.network = dffn(bands, classes, layers_num)
        self.spa_att = HPDM(bands)
    
    def forward(self, x):
        # input: (b, 1, d, w, h)
        x = self.spa_att(x.squeeze(1)).unsqueeze(1)
        x = self.network(x)
        return x 


def DFFN(dataset, bands, classes):
    if dataset == 'PaviaU':
        #(23x23)
        model = dffn(bands, classes, [4,4,4])
    else:
        #IndianPines(25x25) & Salinas(27x27)
        model = dffn(bands, classes, [3,3,3])
    return model


def HPDM_DFFN(dataset, bands, classes):
    if dataset == 'PaviaU':
        #(23x23)
        model = hpdm_dffn(bands, classes, [4,4,4])
    else:
        #IndianPines(25x25) & Salinas(27x27)
        model = hpdm_dffn(bands, classes, [3,3,3])
    return model

