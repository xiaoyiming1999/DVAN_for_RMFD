#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings


# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN_7l(nn.Module):
    def __init__(self, num_cls=10):
        super(CNN_7l, self).__init__()

        # For data with a higher sampling rate, the kernel size of the first-layer convolution must be slightly larger.
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=25, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            )

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout())

        self.layer6 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout())

        self.layer7 = nn.Linear(128, num_cls)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x