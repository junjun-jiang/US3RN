import torch
import torch.nn  as nn
import numpy as np
import torch.nn.functional as F


class Downsample(nn.Module):
    def __init__(self, n_channels, ratio):
        super(Downsample, self).__init__()
        self.ratio = ratio
        dconvs = []
        for i in range(int(np.log2(ratio))):
            dconvs.append(nn.Conv2d(n_channels, n_channels, 3, stride=2, padding=1, dilation=1, groups=n_channels, bias=True))

        self.downsample = nn.Sequential(*dconvs)

    def forward(self,x):
        h = self.downsample(x)
        return h


class Upsample(nn.Module):
    def __init__(self, n_channels, ratio):
        super(Upsample, self).__init__()
        uconvs = []
        for i in range(int(np.log2(ratio))):
            uconvs.append(nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.upsample = nn.Sequential(*uconvs)


    def forward(self,x):
        h = self.upsample(x)
        return h


class resblock(nn.Module):
    def __init__(self, n_channels, kernel_size):
        super(resblock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size, stride=1, padding=kernel_size //2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size, stride=1, padding=kernel_size //2, bias=True),
        )
        self.relu = nn.ReLU()

    def forward(self,x):
        res = self.body(x)
        x = res + x
        return self.relu(x)
