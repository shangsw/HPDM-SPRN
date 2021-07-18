# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import HPDM

# The original implementation of FDSSC is based on Keras. Code is avilable at https://github.com/shuguang-52/FDSSC
# This is a PyTorch version of FDSSC

def _bn_prelu(input_channels):
    layers = nn.Sequential(
        nn.BatchNorm3d(input_channels),
        nn.PReLU()
        )
    return layers


def _spectral_conv(input_channels, out_channels):
    layers = nn.Sequential(
        _bn_prelu(input_channels),
        nn.Conv3d(input_channels, out_channels, (7,1,1), padding=(3, 0, 0))
        )
    return layers


def _spacial_conv(input_channels, out_channels):
    layers = nn.Sequential(
        _bn_prelu(input_channels),
        nn.Conv3d(input_channels, out_channels, (1, 3, 3),padding=(0, 1, 1))
        )
    return layers


class FDSSC(nn.Module):
    def __init__(self, bands, classes):
        super(FDSSC, self).__init__()
        growth_rate = 12
        self.bands = bands
        self.conv1 = nn.Conv3d(1, 24, (7,1,1), padding=0)
        self.spe_conv1 = _spectral_conv(24, growth_rate)
        self.spe_conv2 = _spectral_conv(24+growth_rate, growth_rate)
        self.spe_conv3 = _spectral_conv(24+growth_rate*2, growth_rate)
        self.bn_prelu1 = _bn_prelu(24+growth_rate*3)

        self.size1 = self.bands - 6
        self.tran_conv = nn.Conv3d(24+growth_rate*3, 200, (self.size1,1,1), padding=0)
        self.tran_bn = _bn_prelu(200)
        self.conv2 = nn.Conv3d(1, 24, (200,3,3),padding=0)

        self.spa_conv1 = _spacial_conv(24, growth_rate)
        self.spa_conv2 = _spacial_conv(24+growth_rate, growth_rate)
        self.spa_conv3 = _spacial_conv(24+growth_rate*2, growth_rate)
        self.bn_prelu2 = _bn_prelu(24+growth_rate*3)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(24+growth_rate*3, classes)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, input):
        #dense spectral block
        x1_0 = self.conv1(input)
        x1_1 = self.spe_conv1(x1_0)
        x1_1_ = torch.cat((x1_0, x1_1), dim=1)
        x1_2 = self.spe_conv2(x1_1_)
        x1_2_ = torch.cat((x1_0, x1_1, x1_2), dim=1)
        x1_3 = self.spe_conv3(x1_2_)
        x1 = torch.cat((x1_0, x1_1, x1_2, x1_3), dim=1)
        x1 = self.bn_prelu1(x1)

        # Reducing dimension layer
        tran1 = self.tran_conv(x1)
        tran1 = self.tran_bn(tran1)
        tran2 = tran1.permute(0,2,1,3,4)
        
        x2_0 = self.conv2(tran2)
        #dense spatial block
        x2_1 = self.spa_conv1(x2_0)
        x2_1_ = torch.cat((x2_0, x2_1), dim=1)
        x2_2 = self.spa_conv2(x2_1_)
        x2_2_ = torch.cat((x2_0, x2_1, x2_2), dim=1)
        x2_3 = self.spa_conv3(x2_2_)
        x2 = torch.cat((x2_0, x2_1, x2_2, x2_3), dim=1)
        x2 = self.bn_prelu2(x2)
        #classifier block
        pool1 = self.avgpool(x2)
        drop1 = self.dropout(pool1.flatten(1))
        dense = self.fc(drop1)

        return dense

class HPDM_FDSSC(nn.Module):
    def __init__(self, bands, classes):
        super(HPDM_FDSSC, self).__init__()
        self.network = FDSSC(bands, classes)
        self.spa_att = HPDM(bands)
    
    def forward(self, x):
        # input: (b, 1, d, w, h)
        x = self.spa_att(x.squeeze(1)).unsqueeze(1)
        x = self.network(x)
        return x 