"""
gat.py
======
PURPOSE:
    Graph Attention Network (GAT) model for node-level fraud classification.
    GAT uses learned attention weights to aggregate neighbor features.
    Better than GraphSAGE when some neighbors are more informative than others.

ARCHITECTURE:
    Input (166 features)
        → GATConv layer 1 (heads=8) → ELU → Dropout
        → GATConv layer 2 (heads=1, concat=False) → output logits (2 classes)

INPUTS to forward():
    - x          : node feature tensor (num_nodes x 166)
    - edge_index : graph connectivity tensor (2 x num_edges)

HOW IT WORKS:
    __init__(in_channels, hidden_channels, out_channels, heads, dropout)
        Build two GATConv layers from PyG
        First layer: multi-head (heads=8), concat outputs → hidden_channels * heads
        Second layer: single head, average outputs → out_channels (2)

    forward(x, edge_index)
        Layer 1: GATConv → ELU → Dropout
        Layer 2: GATConv → return logits
        Return: logits tensor (num_nodes x 2)

    get_attention_weights(x, edge_index)
        Run forward pass but also return attention coefficients
        Useful for explainability — which edges got high attention?
        Return: logits, attention_weights

OUTPUT:
    Logits tensor of shape (num_nodes, 2)
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_attention_weights(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index_1, alpha_1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits, (edge_index_2, alpha_2) = self.conv2(x, edge_index, return_attention_weights=True)

        attention_weights = {
            "layer1": (edge_index_1, alpha_1),
            "layer2": (edge_index_2, alpha_2),
        }
        return logits, attention_weights

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
