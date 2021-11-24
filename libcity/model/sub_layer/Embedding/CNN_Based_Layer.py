import random
from decimal import Decimal
from logging import getLogger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from tqdm import tqdm

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class CNN(nn.Module):
    def __init__(self, height, width, n_layers):
        super(CNN, self).__init__()
        self.height = height
        self.width = width
        self.n_layers = n_layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1)))
        for i in range(1, self.n_layers):
            self.conv.append(
                nn.ReLU()
            )
            self.conv.append(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
            )
        self.relu = nn.ReLU()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=self.height * self.width * 16, out_channels=32, kernel_size=(1, 1)),
            nn.ReLU()
        )

    def forward(self, x):
        # (B, T, H, W, H, W)
        x = x.reshape(-1, 1, self.height, self.width)
        # (B * T * H * W, 1, H, W)
        _x = x
        x = self.conv[0](x)
        for i in range(1, self.n_layers):
            x += _x
            x = self.conv[2 * i - 1](x)
            _x = x
            x = self.conv[2 * i](x)
        x += _x
        x = self.relu(x)
        x = x.reshape(-1, self.height * self.width * 16, self.height, self.width)
        # (B * T, H * W * 16, H, W)
        x = self.embed(x)
        # (B * T, 32, H, W)
        return x


class SpatialViewConv(nn.Module):
    def __init__(self, inp_channel, oup_channel, kernel_size, stride=1, padding=0):
        super(SpatialViewConv, self).__init__()
        self.inp_channel = inp_channel
        self.oup_channel = oup_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=inp_channel, out_channels=oup_channel,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(oup_channel)
        self.relu = nn.ReLU()

    def forward(self, inp):
        return self.relu(self.batch(self.conv(inp)))

