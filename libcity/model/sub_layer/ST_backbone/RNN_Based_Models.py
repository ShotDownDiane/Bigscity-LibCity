import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F

import numpy as np
import math


class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_dim, output_dim, bias=True):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.x2h = nn.Linear(input_dim, 3 * output_dim, bias=bias)
        self.h2h = nn.Linear(output_dim, 3 * output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x,hidden):

        #x.shape=(batch,num_nodes,input_dim)
        #hy.shape=(batch,num_nodes,output_dim)

        batch,num_nodes=x.shape[:2]

        x=self.mode(x)
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        hy=hy.view(batch,num_nodes,-1)

        return hy


class TemproalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim,bias=True):
        super(TemproalGRU, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.bias=bias
        self.gru_cell = GRUCell(input_dim, hidden_dim)


    def forward(self, x):

        # x.shape=(batch,num_timesteps,num_nodes,input_dim)
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        hn = h0[0, :, :]
        x=x.permute(1,0,2,3)  # (num_timesteps, batch,num_nodes,input_dim)

        for seq in range(x.size(0)):
            hn = self.gru_cell(x[seq], hn)
            outs.append(hn)

        outs=torch.vstack(outs)
        outs=outs.permute(1,0,2,3)#(batch,num_timesteps,num_nodes,input_dim)

        return outs

#fixme hidden dim (batch,num_nodes,output_dim)

class LSTMCell(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        #x.shape=(batch,num_nodes,input_dim)
        #hidden.shape=(2,batch,num_nodes,input_dim)


        hx, cx = hidden

        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)#(batch*num_nodes,4*hidden)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)#(batch*num_nodes,hidden)

        #fixme shape
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))



        return (hy, cy)



class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(TemporalLSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # x.shape=(batch,num_timesteps,num_nodes,input_dim)
        if torch.cuda.is_available():
            hn = Variable(torch.zeros(x.size(0),x.size(2), self.hidden_dim).cuda())
        else:
            hn = Variable(torch.zeros(x.size(0),x.size(2), self.hidden_dim))

        # Initialize cell state
        if torch.cuda.is_available():
            cn = Variable(torch.zeros(x.size(0),x.size(2), self.hidden_dim).cuda())
        else:
            cn = Variable(torch.zeros(x.size(0),x.size(2), self.hidden_dim))

        outs = []

        # cn = c0[0, :, :]
        # hn = h0[0, :, :]
        x=x.permute(1,0,2,3)
        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[seq], (hn, cn))
            outs.append(hn)

        outs=torch.vstack(outs)
        outs=outs.permute(1,0,2,3)
        # out.shape=(batch,num_timesteps,num_nodes,output_dim)
        return outs
