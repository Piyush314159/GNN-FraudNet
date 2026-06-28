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

TODO: Implement GAT class below
"""

class GAT:
    # TODO: implement __init__, forward, get_attention_weights
    pass
