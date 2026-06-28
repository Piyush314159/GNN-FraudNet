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

TODO: Implement GraphSAGE class below
"""

class GraphSAGE:
    # TODO: implement __init__, forward, reset_parameters
    pass
