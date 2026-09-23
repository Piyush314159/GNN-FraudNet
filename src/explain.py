"""
explain.py
==========
PURPOSE:
    Provide explainability for GNN predictions.
    Answer: WHY was this transaction flagged as fraud?

TWO APPROACHES:
    1. GNNExplainer (PyG built-in) — explains individual node predictions
       by finding the subgraph and features most important for that prediction
    2. SHAP — approximate feature importance across all nodes

INPUTS:
    - model    : trained GraphSAGE or GAT model
    - data     : PyG Data object
    - node_idx : specific node to explain (for GNNExplainer)

HOW IT WORKS:
    run_gnn_explainer(model, data, node_idx)
        Initialize PyG GNNExplainer with trained model
        Run explanation for node_idx
        Returns: node_mask (which features mattered), edge_mask (which edges mattered)

    plot_explanation_subgraph(data, node_idx, edge_mask)
        Extract top-k important edges around node_idx
        Draw subgraph using networkx
        Color nodes: red=illicit, green=licit, gray=unknown, blue=node_idx itself
        Save to results/plots/explanation_node_{node_idx}.png

    run_shap(model, data, node_indices)
        Use SHAP KernelExplainer (model-agnostic) on the given test nodes
        Only the target node's own features are perturbed; the rest of the
        graph (and its neighbors' features) is held fixed for each explanation
        Return: shap_values array of shape (len(node_indices), num_features)

    plot_shap_summary(shap_values, feature_names)
        Bar chart of top 15 most important features by mean |SHAP|
        Save to results/plots/shap_summary.png

OUTPUT:
    Explanation plots in results/plots/
    Edge mask and feature mask tensors

NOTE:
    run_shap() re-runs a full forward pass of the GNN per SHAP sample, so it
    is intentionally limited to a small number of nodes/samples by default.
"""
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

PLOT_DIR = "results/plots"

_CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c", -1: "#95a5a6"}
_CENTER_COLOR = "#3498db"


def run_gnn_explainer(model, data, node_idx, epochs=200, lr=0.01):
    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
    )
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    return explanation.node_mask, explanation.edge_mask


def plot_explanation_subgraph(data, node_idx, edge_mask, num_hops=2, top_k=15):
    os.makedirs(PLOT_DIR, exist_ok=True)
    node_idx = int(node_idx)

    subset, sub_edge_index, mapping, edge_hop_mask = k_hop_subgraph(
        node_idx, num_hops, data.edge_index, relabel_nodes=True
    )
    sub_weights = edge_mask[edge_hop_mask].detach().cpu().numpy()

    if len(sub_weights) > top_k:
        keep = np.argsort(sub_weights)[-top_k:]
    else:
        keep = np.arange(len(sub_weights))

    center_local = int(mapping[0].item())

    graph = nx.DiGraph()
    for local_idx, global_idx in enumerate(subset.tolist()):
        graph.add_node(local_idx, color=_CLASS_COLORS.get(data.y[global_idx].item(), "#95a5a6"))

    src, dst = sub_edge_index[0].tolist(), sub_edge_index[1].tolist()
    for i in keep:
        graph.add_edge(src[i], dst[i], weight=float(sub_weights[i]))

    node_colors = [
        _CENTER_COLOR if n == center_local else graph.nodes[n]["color"] for n in graph.nodes()
    ]
    edge_widths = [graph[u][v]["weight"] * 3 + 0.5 for u, v in graph.edges()]

    pos = nx.spring_layout(graph, seed=42)

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=250, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.5, arrows=True, width=edge_widths, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)
    ax.set_title(f"GNNExplainer Subgraph — Node {node_idx}", fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    save_path = os.path.join(PLOT_DIR, f"explanation_node_{node_idx}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_shap(model, data, node_indices=None, n_explain=20, n_background=30, nsamples=50):
    import shap

    model.eval()
    x_np = data.x.detach().cpu().numpy()
    device = data.x.device

    if node_indices is None:
        test_idx = data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        node_indices = test_idx[:n_explain]
    else:
        node_indices = np.asarray(node_indices)

    rng = np.random.default_rng(42)
    background_pool = rng.choice(x_np.shape[0], size=min(n_background, x_np.shape[0]), replace=False)
    background = x_np[background_pool]

    all_shap_values = []
    for node_id in node_indices:
        node_id = int(node_id)

        def predict_fn(feature_batch, node_id=node_id):
            probs = np.zeros(feature_batch.shape[0])
            with torch.no_grad():
                for i, row in enumerate(feature_batch):
                    x_mod = x_np.copy()
                    x_mod[node_id] = row
                    x_tensor = torch.tensor(x_mod, dtype=torch.float, device=device)
                    logits = model(x_tensor, data.edge_index)
                    probs[i] = torch.softmax(logits[node_id], dim=0)[1].item()
            return probs

        explainer = shap.KernelExplainer(predict_fn, background)
        sv = explainer.shap_values(x_np[node_id:node_id + 1], nsamples=nsamples)
        all_shap_values.append(np.asarray(sv).reshape(-1))

    return np.array(all_shap_values)


def plot_shap_summary(shap_values, feature_names, top_k=15):
    os.makedirs(PLOT_DIR, exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[-top_k:]
    feature_names = np.array(feature_names)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(feature_names[order], mean_abs[order], color="#9b59b6", edgecolor="white")
    ax.set_title(f"Top {top_k} Features by Mean |SHAP Value|", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()

    save_path = os.path.join(PLOT_DIR, "shap_summary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
