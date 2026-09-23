"""
graph_builder.py
================
PURPOSE:
    Take the raw PyG Data object and prepare it for training —
    normalize features, create train/val/test masks, compute class weights.

INPUTS:
    - PyG Data object from data_loader.build_pyg_data()

HOW IT WORKS:
    Step 1 — create_masks(data)
        Filter out unknown nodes (y == -1)
        Split labeled nodes 70% train / 15% val / 15% test
        Set data.train_mask, data.val_mask, data.test_mask as boolean tensors
        Return: updated Data object

    Step 2 — normalize_features(data)
        Must be called AFTER create_masks(), since it needs data.train_mask
        Compute mean and std over training nodes only (avoid data leakage)
        Apply: x = (x - mean) / std to ALL nodes
        Return: Data object with normalized x

    Step 3 — get_class_weights(data)
        Count illicit vs licit in training set
        Compute weight = total / (2 * count_per_class)
        Return: tensor of shape (2,) for use in weighted CrossEntropyLoss

    Step 4 — inspect_graph(data)
        Print summary:
            - Total nodes, total edges
            - Labeled vs unknown nodes
            - Illicit vs licit ratio in train set
        Return: None (print only)

OUTPUT:
    PyG Data object with .train_mask, .val_mask, .test_mask, normalized .x
    Class weight tensor
"""
import numpy as np
import torch


def normalize_features(data):
    """Z-score normalize features using training node statistics only."""
    x = data.x.clone()
    train_mask = data.train_mask

    train_x = x[train_mask]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0)
    std[std == 0] = 1.0

    data.x = (x - mean) / std
    return data


def create_masks(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create boolean train/val/test masks on labeled nodes (y != -1)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_nodes = data.y.size(0)

    labeled_mask = data.y != -1
    labeled_indices = labeled_mask.nonzero(as_tuple=True)[0].numpy()
    np.random.shuffle(labeled_indices)

    n_labeled = len(labeled_indices)
    n_train = int(n_labeled * train_ratio)
    n_val = int(n_labeled * val_ratio)

    train_idx = labeled_indices[:n_train]
    val_idx = labeled_indices[n_train:n_train + n_val]
    test_idx = labeled_indices[n_train + n_val:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data


def get_class_weights(data):
    """Inverse-frequency class weights [w_licit, w_illicit] from the training split."""
    train_labels = data.y[data.train_mask]
    n_licit = (train_labels == 0).sum().item()
    n_illicit = (train_labels == 1).sum().item()
    n_total = n_licit + n_illicit

    w_licit = n_total / (2.0 * n_licit)
    w_illicit = n_total / (2.0 * n_illicit)

    return torch.tensor([w_licit, w_illicit], dtype=torch.float)


def inspect_graph(data):
    """Print graph/label/mask summary statistics."""
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    num_features = data.x.size(1)

    labeled = (data.y != -1).sum().item()
    unknown = (data.y == -1).sum().item()
    illicit = (data.y == 1).sum().item()
    licit = (data.y == 0).sum().item()

    src_nodes = data.edge_index[0]
    dst_nodes = data.edge_index[1]
    out_deg = torch.zeros(num_nodes, dtype=torch.long)
    in_deg = torch.zeros(num_nodes, dtype=torch.long)
    out_deg.scatter_add_(0, src_nodes, torch.ones_like(src_nodes))
    in_deg.scatter_add_(0, dst_nodes, torch.ones_like(dst_nodes))
    total_deg = in_deg + out_deg

    print("=" * 60)
    print("  GRAPH SUMMARY")
    print("=" * 60)
    print(f"  Nodes              : {num_nodes:,}")
    print(f"  Edges (directed)   : {num_edges:,}")
    print(f"  Features per node  : {num_features}")
    print(f"  Avg degree         : {total_deg.float().mean():.2f}")
    print(f"  Max degree         : {total_deg.max().item()}")
    print(f"  Isolated nodes     : {(total_deg == 0).sum().item():,}")
    print()
    print(f"  Labeled nodes      : {labeled:,} ({labeled / num_nodes * 100:.1f}%)")
    print(f"    Illicit (fraud)  : {illicit:,}")
    print(f"    Licit (legit)    : {licit:,}")
    print(f"    Ratio (licit:ill): {licit / max(illicit, 1):.1f} : 1")
    print(f"  Unknown nodes      : {unknown:,} ({unknown / num_nodes * 100:.1f}%)")

    if hasattr(data, "train_mask"):
        print()
        print(f"  Train mask         : {data.train_mask.sum().item():,}")
        print(f"  Val mask           : {data.val_mask.sum().item():,}")
        print(f"  Test mask          : {data.test_mask.sum().item():,}")
    print("=" * 60)
