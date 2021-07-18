# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import HPDM

# Original implementation of BASSNet is based on Torch. Code is available at https://github.com/hbutsuak95/BASS-Net
# This is a PyTorch version of BASS-Net
# We add BN to speed up the training process

class BASS(nn.Module):
    def __init__(self, bands, classes, patch_size, nbands, block1_conv1):
        super(BASS, self).__init__()
        assert block1_conv1 % nbands == 0, 'bands should be divided by nbands'

        self.nbands = nbands
        self.patch_size = patch_size
        self.band_size = block1_conv1 // nbands
        self.conv0 = nn.Conv2d(bands, block1_conv1, (1,1))
        self.bn0 = nn.BatchNorm2d(block1_conv1)

        self.conv1_1 = nn.Conv1d(patch_size*patch_size, 20, kernel_size=3, padding=0)
        self.bn1_1 = nn.BatchNorm1d(20)
        self.conv1_2 = nn.Conv1d(20, 20, kernel_size=3, padding=0)
        self.bn1_2 = nn.BatchNorm1d(20)
        self.conv1_3 = nn.Conv1d(20, 10, kernel_size=3, padding=0)
        self.bn1_3 = nn.BatchNorm1d(10)
        self.conv1_4 = nn.Conv1d(10, 5, kernel_size=5, padding=0)
        self.bn1_4 = nn.BatchNorm1d(5)

        self.fc1 = nn.Linear(nbands*(self.band_size-10)*5, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = F.relu(self.bn0(self.conv0(x)))

        spx = torch.split(x, self.band_size, 1) #spx[i]:(batch, band_size, patch_size, patch_size)
        for i in range(self.nbands):
            x1 = spx[i].view(-1, self.band_size, self.patch_size*self.patch_size).permute(0,2,1)
            x1 = F.relu(self.bn1_1(self.conv1_1(x1)))
            x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
            x1 = F.relu(self.bn1_3(self.conv1_3(x1)))
            x1 = F.relu(self.bn1_4(self.conv1_4(x1)))
            if i == 0:
                x = x1
            else:
                x = torch.cat((x,x1), 1)
        
        x = F.relu(self.fc1(x.flatten(1)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class HPDM_BASS(nn.Module):
    def __init__(self, bands, classes, patch_size, nbands, block1_conv1):
        super(HPDM_BASS, self).__init__()
        self.network = BASS(bands, classes, patch_size, nbands, block1_conv1)
        self.spa_att = HPDM(bands)
    
    def forward(self, x):
        # input: (b, 1, d, w, h)
        x = self.spa_att(x.squeeze(1)).unsqueeze(1)
        x = self.network(x)
        return x 

def BASSNet(dataset, bands, classes, patch_size):
    if dataset == 'PaviaU':
        model = BASS(bands, classes, patch_size, 5, 100)
    elif dataset == 'IndianPines':
        model = BASS(bands, classes, patch_size, 10, 220)
    elif dataset == 'Salinas':
        model = BASS(bands, classes, patch_size, 14, 224)  
    return model

def HPDM_BASSNet(dataset, bands, classes, patch_size):
    if dataset == 'PaviaU':
        model = HPDM_BASS(bands, classes, patch_size, 5, 100)
    elif dataset == 'IndianPines':
        model = HPDM_BASS(bands, classes, patch_size, 10, 220)
    elif dataset == 'Salinas':
        model = HPDM_BASS(bands, classes, patch_size, 14, 224)  
    return model