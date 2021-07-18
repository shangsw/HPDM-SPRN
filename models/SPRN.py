import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
from utils import HPDM

# A PyTorch implementation of SPRN (Spectral Partitioning Residual Network)

class Res2(nn.Module):  
    def __init__(self, in_channels, inter_channels, kernel_size, padding=0):
        super(Res2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        return X


class Res(nn.Module):  
    def __init__(self, in_channels, kernel_size, padding, groups=1):
        super(Res, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.res2 = Res2(in_channels, 32, kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Z = self.res2(X)
        return F.relu(X + Y + Z)


class sprn(nn.Module):
    def __init__(self, bands, classes, groups, groups_width, spa=False):
        super(sprn, self).__init__()
        self.bands = bands
        self.classes = classes
        self.spa = spa
        fc_planes = 128

        # pad the bands with final values
        new_bands = math.ceil(bands/groups) * groups
        pad_size = new_bands - bands
        self.pad = nn.ReplicationPad3d((0,0,0,0,0,pad_size))
        
        # HPDM
        if self.spa:
            self.spa_att = HPDM(new_bands) 
        
        # SPRN
        self.conv1 = nn.Conv2d(new_bands, groups*groups_width, (1,1), groups=groups)
        self.bn1 = nn.BatchNorm2d(groups*groups_width)

        self.res0 = Res(groups*groups_width, (1,1), (0,0), groups=groups)
        self.res1 = Res(groups*groups_width, (1,1), (0,0), groups=groups)
        
        self.conv2 = nn.Conv2d(groups_width*groups, fc_planes, (1,1))
        self.bn2 = nn.BatchNorm2d(fc_planes)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.out_fc = nn.Linear(fc_planes, classes)

    def forward(self, x):
        # input: (b, 1, d, w, h)
        x = self.pad(x).squeeze(1)
        if self.spa:
            x = self.spa_att(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res0(x)
        x = self.res1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x).flatten(1)
        x = self.out_fc(x)
        return x

def SPRN(dataset, bands, classes):
    if dataset == 'PaviaU':
        model = sprn(bands, classes, 5, 64)
    elif dataset == 'IndianPines':
        model = sprn(bands, classes, 11, 37)
    elif dataset == 'Salinas':
        model = sprn(bands, classes, 11, 37)
    return model

def HPDM_SPRN(dataset, bands, classes):
    if dataset == 'PaviaU':
        model = sprn(bands, classes, 5, 64, spa=True)
    elif dataset == 'IndianPines':
        model = sprn(bands, classes, 11, 37, spa=True)
    elif dataset == 'Salinas':
        model = sprn(bands, classes, 11, 37, spa=True)
    return model