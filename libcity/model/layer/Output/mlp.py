import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#fixme 明确n,h,w的指代
class FusionLayer(nn.Module):
    # Matrix-based fusion
    def __init__(self, n, h, w, device):
        super(FusionLayer, self).__init__()
        # define the trainable parameter
        self.weights = nn.Parameter(torch.FloatTensor(1, n, h, w).to(device))

    def forward(self, x):
        # assuming x is of size B-n-h-w
        x = x * self.weights  # element-wise multiplication
        return x