import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np
import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


#fixme 如何定义cell
class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.input_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        Args:
            inputs: shape (batch_size, self.num_nodes * self.input_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.hidden_state_size) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)

        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state  # 循环
        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.output_dim = config.get('output_dim', 1)
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.output_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        Args:
            inputs:  shape (batch_size, self.num_nodes * self.output_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.num_nodes * self.output_dim) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)