"""
graphsage.py
============
PURPOSE:
    GraphSAGE model for node-level fraud classification.
    GraphSAGE aggregates neighbor features by sampling + mean/max pooling.

ARCHITECTURE:
    Input (166 features)
        → SAGEConv layer 1 → BatchNorm → ReLU → Dropout
        → SAGEConv layer 2 → BatchNorm → ReLU → Dropout
        → Linear layer → output logits (2 classes)

INPUTS to forward():
    - x          : node feature tensor (num_nodes x 166)
    - edge_index : graph connectivity tensor (2 x num_edges)

HOW IT WORKS:
    __init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        Build stack of SAGEConv layers from PyG
        Add BatchNorm after each conv layer
        Add final linear classifier

    forward(x, edge_index)
        Pass through each SAGEConv layer
        Apply BatchNorm → ReLU → Dropout after each layer
        Final linear layer gives logits
        Return: logits tensor (num_nodes x 2)

    reset_parameters()
        Re-initialize all weights (useful for repeated experiments)

OUTPUT:
    Logits tensor of shape (num_nodes, 2)
    Apply softmax externally for probabilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.classifier.reset_parameters()
