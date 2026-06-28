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
        Returns: edge_mask (which edges mattered), feature_mask (which features mattered)

    plot_explanation_subgraph(data, node_idx, edge_mask)
        Extract top-k important edges around node_idx
        Draw subgraph using networkx
        Color nodes: red=illicit, green=licit, gray=unknown
        Save to results/plots/explanation_node_{node_idx}.png

    run_shap(model, data)
        Use SHAP KernelExplainer (model-agnostic) on test nodes
        Compute SHAP values for each of 166 features
        Return: shap_values array

    plot_shap_summary(shap_values, feature_names)
        Bar chart of top 15 most important features by mean |SHAP|
        Save to results/plots/shap_summary.png

OUTPUT:
    Explanation plots in results/plots/
    Edge mask and feature mask tensors

TODO: Implement all functions below
"""

def run_gnn_explainer(model, data, node_idx):
    pass

def plot_explanation_subgraph(data, node_idx, edge_mask):
    pass

def run_shap(model, data):
    pass

def plot_shap_summary(shap_values, feature_names):
    pass
