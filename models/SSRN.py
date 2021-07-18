# -*- coding: utf-8 -*-
import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
from utils import HPDM

# Original implementation of SSRN is based on Keras. Code is available at https://github.com/zilongzhong/SSRN
# This PyTorch version of SSRN is modified from the project https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network 

class Residual(nn.Module):  
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                                    kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 
                                    kernel_size=kernel_size, padding=padding,stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

class SSRN(nn.Module):
    #SSRN network
    def __init__(self, bands, classes):
        super(SSRN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(7, 1, 1), stride=(2, 1, 1))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.res_net1 = Residual(24, 24, (7, 1, 1), (3, 0, 0))
        self.res_net2 = Residual(24, 24, (7, 1, 1), (3, 0, 0))
        self.res_net3 = Residual(24, 24, (1, 3, 3), (0, 1, 1))
        self.res_net4 = Residual(24, 24, (1, 3, 3), (0, 1, 1))

        kernel_3d = math.ceil((bands - 6) / 2)

        self.conv2 = nn.Conv3d(in_channels=24, out_channels=128, padding=(0, 0, 0),
                               kernel_size=(kernel_3d, 1, 1), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0),
                               kernel_size=(128, 3, 3), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.avg_pooling = nn.AdaptiveAvgPool3d((1,1,1))
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(24, classes)
        )

    def forward(self, X):
        # input: (b, 1, d, w, h)
        x1 = self.batch_norm1(self.conv1(X))

        x2 = self.res_net1(x1)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 2, 1, 3, 4)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        x = self.full_connection(x4)
        return x

class HPDM_SSRN(nn.Module):
    def __init__(self, bands, classes):
        super(HPDM_SSRN, self).__init__()
        self.network = SSRN(bands, classes)
        self.spa_att = HPDM(bands)
    
    def forward(self, x):
        # input: (b, 1, d, w, h)
        x = self.spa_att(x.squeeze(1)).unsqueeze(1)
        x = self.network(x)
        return x 