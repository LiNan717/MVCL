import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class ResidualGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, K, dropout=0.5, with_identity=True):
        super(ResidualGCNLayer, self).__init__()
        self.cheb_conv = ChebConv(in_channels, out_channels, K)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.with_identity = with_identity
        self.match_dim = nn.Linear(in_channels, out_channels)

    def forward(self, edge_index, x):
        identity = x
        identity = self.match_dim(identity)
        x = self.cheb_conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        if self.with_identity:
            x = x + identity
        return x


class DeepResidualGCN(nn.Module):
    def __init__(self, num_features, hidden_channels, output_channels, num_layers, K=3):
        super(DeepResidualGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # Input layer
        self.layers.append(ResidualGCNLayer(num_features, hidden_channels, K))

        # Hidden layers
        for i in range(num_layers - 2):
            self.layers.append(ResidualGCNLayer(hidden_channels, hidden_channels, K))

        # Output layer
        self.layers.append(ResidualGCNLayer(hidden_channels, output_channels, K, with_identity=False))

    def forward(self, edge_index, x):
        for i, layer in enumerate(self.layers):
            x = layer(edge_index, x)
        return F.log_softmax(x, dim=1)
