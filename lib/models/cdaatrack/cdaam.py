import torch
import torch.nn as nn
import numpy as np
import cv2

# Color-Depth Aware Attention Module
class CDAAM(nn.Module):
    def __init__(self,inchannel, outchannel):
        super(CDAAM, self).__init__()
        self.conv0 = nn.Conv2d(inchannel, outchannel, kernel_size=16, stride=16, padding=0)  # for foreground
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=16, stride=16, padding=0)  # for background
        # self.conv2 = nn.Conv2d(inchannel, outchannel, kernel_size=16, stride=16, padding=0)

    def forward(self, x, priors):

        x0 = self.conv0(priors[:, 0, :, :].unsqueeze(1)) * x         # for foreground
        x1 = self.conv1(priors[:, 1, :, :].unsqueeze(1)) * x         # for background
        # x2=self.conv2(priors[:,2,:,:].unsqueeze(1)) * x
        x = x + x0 - x1
        return x

