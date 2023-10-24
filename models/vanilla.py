import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Creates a NN using nn.ModuleList to automatically adjust the number of layers.
    For each hidden layer, the number of inputs and outputs is constant.

    Inputs:
        in_dim (int):               number of features contained in the input layer.
        out_dim (int):              number of features input and output from each hidden layer,
                                    including the output layer.
        num_layers (int):           number of layers in the network
        activation (torch function): activation function to be used during the hidden layers
    """

    def __init__(self, in_dim, out_dim, num_layers, activation=torch.nn.ReLU(), layer_norm=False, batch_norm=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        h_dim = in_dim if out_dim < 10 else out_dim

        # create the input layer
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(nn.Linear(in_dim, h_dim))
            else:
                self.layers.append(nn.Linear(h_dim, h_dim))
            if layer_norm: self.layers.append(nn.LayerNorm(h_dim))
            if batch_norm: self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(activation)
        self.layers.append(nn.Linear(h_dim, out_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

