# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from utils import ConvOffset2D, HPDM
import torch

# Original implementation of DHCNet is based on Tensorflow. Code is available at https://github.com/ordinarycore/DHCNet
# This is a PyTorch version of DHCNet

class DHCNet(nn.Module):

    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are initialized with xavier_initializer
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, bands, classes):
        super(DHCNet, self).__init__()
        self.conv1 = nn.Conv2d(bands, 96, (3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, (3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        # self.pool1 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(96, 108, (3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(108)
        self.conv4 = nn.Conv2d(108, 108, (3,3), padding=1)
        self.bn4 = nn.BatchNorm2d(108)
        self.offset4 = ConvOffset2D(108)

        # self.conv_pool = nn.Conv2d(108, 108, (2,2), padding=0, stride=2)
        # self.conv_bn = nn.BatchNorm2d(108)

        self.conv5 = nn.Conv2d(108, 128, (3,3), padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.offset5 = ConvOffset2D(128)
        self.conv6 = nn.Conv2d(128, 128, (3,3), padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 200)
        self.bn_fc = nn.BatchNorm1d(200)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, classes)

        self.apply(self.weight_init)

    def forward(self, input):
        input = input.squeeze(1)
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.offset4(x)
        # x = F.relu(self.conv_bn(self.conv_pool(x)))

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.offset5(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.avgpool(x)

        x = x.flatten(1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class HPDM_DHCNet(nn.Module):
    def __init__(self, bands, classes):
        super(HPDM_DHCNet, self).__init__()
        self.network = DHCNet(bands, classes)
        self.spa_att = HPDM(bands)
    
    def forward(self, x):
        # input: (b, 1, d, w, h)
        x = self.spa_att(x.squeeze(1)).unsqueeze(1)
        x = self.network(x)
        return x 