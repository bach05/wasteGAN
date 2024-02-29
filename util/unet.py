
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from torchvision.transforms.functional import crop


from skimage.filters import gaussian
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt

import numpy as np
from util.externalTools import DeformableConv2d

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, dilation=1, kernel_size=3, padding=1):
        super().__init__()
        dropout_rate = 0.2
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            #nn.BatchNorm2d(mid_channels),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(dropout_rate),
            nn.GroupNorm(1, mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            #DeformableConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        if self.residual:
            return F.leaky_relu(x + self.double_conv(x), 0.2)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dilation=1, kernel_size=3, padding=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True, dilation=dilation, kernel_size=kernel_size, padding=padding),
            DoubleConv(in_channels, out_channels, in_channels // 2, dilation=dilation, kernel_size=kernel_size, padding=padding),
        )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True, dilation=dilation, kernel_size=kernel_size, padding=padding),
            DoubleConv(in_channels, out_channels, in_channels // 2, dilation=dilation, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class FusionLayer(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        filter_sizes = [16, 8, 8, 16]
        kernel_sizes = [5, 3, 3, 5]
        strides = [2, 2, 2, 2]

        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=3, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(filter_sizes[3], 1, kernel_size=2)
        )
        for m in self.fusion:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)


    def forward(self, x):
        return self.fusion(x)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
