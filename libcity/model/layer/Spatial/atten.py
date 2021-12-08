import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#fixme 将num_of_vertics改成num of nodes
class SpatialAttention(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(SpatialAttention, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(device))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(device))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(device))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(device))

    def forward(self, x):
        """
        Args:
            x(torch.tensor): (B, N, F_in, T)

        Returns:
            torch.tensor: (B,N,N)
        """
        # x * W1 --> (B,N,F,T)(T)->(B,N,F)
        # x * W1 * W2 --> (B,N,F)(F,T)->(B,N,T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        # (W3 * x) ^ T --> (F)(B,N,F,T)->(B,N,T)-->(B,T,N)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        # x = lhs * rhs --> (B,N,T)(B,T,N) -> (B, N, N)
        product = torch.matmul(lhs, rhs)
        # S = Vs * sig(x + bias) --> (N,N)(B,N,N)->(B,N,N)
        s = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        # softmax (B,N,N)
        s_normalized = F.softmax(s, dim=1)
        return s_normalized