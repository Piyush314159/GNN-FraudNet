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
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def load_features(path):
    # TODO: read CSV, drop txId, return float DataFrame
    df = pd.read_csv(path)
    df = df.drop(columns=["txId"])
    return df.astype(float)

def load_edges(path, node_mapping):
    # TODO: read edgelist, convert txIds to integer indices using node_mapping
    # return src_array, dst_array
    df = pd.read_csv(path)
    df["txId1"] = df["txId1"].map(node_mapping)
    df["txId2"] = df["txId2"].map(node_mapping)

    n_before = len(df)
    df = df.dropna(subset = ["txId1", "txId2"])

    if len(df) < n_before:
        print(f"load_edges: dropped {n_before - len(df)} edges with unmapped txId")
    
    src_array = df["txId1"].to_numpy(dtype= np.int64)
    dst_array = df["txId2"].to_numpy(dtype = np.int64)

    return src_array, dst_array

def load_labels(path, node_mapping):
    # TODO: read classes CSV, map labels to 1/0/-1
    # return label array aligned with node_mapping order
    df = pd.read_csv(path)  # columns: txId, class
    label_map = {"1": 1, "2": 0, "unknown": -1}
    df["class"] = df["class"].astype(str).map(label_map)

    # build array in node_mapping's index order
    num_nodes = len(node_mapping)
    labels = np.full(num_nodes, -1, dtype=np.int64)
    for tx_id, cls in zip(df["txId"], df["class"]):
        idx = node_mapping.get(tx_id)
        if idx is not None:
            labels[idx] = cls
    return labels

def build_pyg_data(raw_dir, save_dir):
    # TODO: call above three functions
    # build PyG Data(x, edge_index, y)
    # save to save_dir
    # return Data object
    features_path = os.path.join(raw_dir, "elliptic_txs_features.csv")
    edges_path = os.path.join(raw_dir, "elliptic_txs_edgelist.csv")
    classes_path = os.path.join(raw_dir, "elliptic_txs_classes.csv")

    # node_mapping needs txId in original row order — read it once here,
    # since load_features() drops the column before returning
    raw_features = pd.read_csv(features_path)
    node_mapping = {tx_id: i for i, tx_id in enumerate(raw_features["txId"])}

    x_df = load_features(features_path)
    src_array, dst_array = load_edges(edges_path, node_mapping)
    y_array = load_labels(classes_path, node_mapping)

    x = torch.tensor(x_df.values, dtype=torch.float)
    edge_index = torch.tensor(np.stack([src_array, dst_array]), dtype=torch.long)
    y = torch.tensor(y_array, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "elliptic_graph.pt")
    torch.save(data, save_path)
    print(f"Saved PyG Data object to {save_path}")

    return data
