"""
graph_builder.py
================
PURPOSE:
    Take the raw PyG Data object and prepare it for training —
    normalize features, create train/val/test masks, compute class weights.

INPUTS:
    - PyG Data object from data_loader.build_pyg_data()

HOW IT WORKS:
    Step 1 — normalize_features(data)
        Only normalize labeled nodes (y != -1)
        Compute mean and std over training nodes only (avoid data leakage)
        Apply: x = (x - mean) / std
        Return: Data object with normalized x

    Step 2 — create_masks(data)
        Filter out unknown nodes (y == -1)
        Split labeled nodes 70% train / 15% val / 15% test
        Set data.train_mask, data.val_mask, data.test_mask as boolean tensors
        Return: updated Data object

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

TODO: Implement each function below
"""

def normalize_features(data):
    # TODO: normalize using train node stats only
    pass

def create_masks(data, train_ratio=0.7, val_ratio=0.15):
    # TODO: split labeled nodes, set boolean masks on data object
    pass

def get_class_weights(data):
    # TODO: compute inverse frequency weights for imbalanced classes
    pass

def inspect_graph(data):
    # TODO: print graph statistics
    pass
