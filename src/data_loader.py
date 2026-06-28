"""
data_loader.py
==============
PURPOSE:
    Load the three raw Elliptic Bitcoin dataset CSVs and combine them
    into a PyTorch Geometric (PyG) Data object ready for GNN training.

INPUTS:
    - data/raw/elliptic_txs_features.csv   → 203k rows x 167 cols (txId + 166 features)
    - data/raw/elliptic_txs_edgelist.csv   → directed edges between transaction nodes
    - data/raw/elliptic_txs_classes.csv    → labels: 1=illicit, 2=licit, unknown=unknown

HOW IT WORKS:
    Step 1 — load_features()
        Read elliptic_txs_features.csv
        Drop txId column, keep 166 float features
        Return: DataFrame of shape (203769, 166)

    Step 2 — load_edges()
        Read elliptic_txs_edgelist.csv
        Map txId strings to integer node indices
        Return: two arrays [src_nodes], [dst_nodes]

    Step 3 — load_labels()
        Read elliptic_txs_classes.csv
        Map: "1" → 1 (illicit/fraud), "2" → 0 (licit/legit), "unknown" → -1
        Return: array of shape (203769,)

    Step 4 — build_pyg_data()
        Call all three above
        Construct PyG Data object:
            data.x          = node feature tensor (203769 x 166)
            data.edge_index = edge index tensor (2 x num_edges)
            data.y          = label tensor (203769,)
        Save to data/processed/elliptic_graph.pt
        Return: PyG Data object

OUTPUT:
    PyG Data object saved at data/processed/elliptic_graph.pt

TODO: Implement each function below
"""

def load_features(path):
    # TODO: read CSV, drop txId, return float DataFrame
    pass

def load_edges(path, node_mapping):
    # TODO: read edgelist, convert txIds to integer indices using node_mapping
    # return src_array, dst_array
    pass

def load_labels(path, node_mapping):
    # TODO: read classes CSV, map labels to 1/0/-1
    # return label array aligned with node_mapping order
    pass

def build_pyg_data(raw_dir, save_dir):
    # TODO: call above three functions
    # build PyG Data(x, edge_index, y)
    # save to save_dir
    # return Data object
    pass
