# -*- coding: utf-8 -*-
import torch
import torch.nn as nn 
import torch.nn.functional as F

# Homogeneous pixel detection module (HPDM)

class HPDM(nn.Module):
    def __init__(self, in_channels):
        super(HPDM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
            
        self.theta1 = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=(1,1), padding=0)
        self.bn = nn.BatchNorm2d(self.inter_channels)
        nn.init.constant_(self.theta1.weight, 0.0)
        nn.init.constant_(self.theta1.bias, 0.0)

    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return z: (b, c, h, w)
        '''
        batch_size, c, w, h = x.size()

        theta_x = F.sigmoid(self.bn(self.theta1(x))) #(b,c//2,w,h)
        center = theta_x[:, :, w//2:w//2+1, h//2:h//2+1] #(b,c//2,1,1)

        norm_center = torch.norm(center, p=2, dim=1)  #(b,1,1)
        norm_x = torch.norm(theta_x, p=2, dim=1)  #(b,w,h)

        center = center.view(batch_size, self.inter_channels, -1).permute(0,2,1)   #(b,1,c//2)
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)    #(b,c//2,w*h)

        f = torch.matmul(center, theta_x)   #(b,1,w*h)
        ab = norm_x * norm_center  #(b,w,h)
        ab = ab.view(batch_size, 1, -1) #(b,1,w*h)
        cos_theta = torch.div(f, ab)

        y = cos_theta.view(batch_size, 1, *x.size()[2:])    #mask:(b,1,w,h)
        z = y * x
        return z