"""
evaluate.py
===========
PURPOSE:
    Compute evaluation metrics for all models and produce comparison table.
    Also generates and saves evaluation plots.

INPUTS:
    - y_true  : true labels (test set)
    - y_pred  : predicted class labels
    - y_prob  : predicted probabilities for illicit class

WHY THESE METRICS:
    Accuracy is misleading on imbalanced data (only 2% fraud).
    We use F1 and PR-AUC which focus on the minority (fraud) class.

HOW IT WORKS:
    compute_f1(y_true, y_pred)
        Compute F1 score for illicit class only
        Return: float

    compute_pr_auc(y_true, y_prob)
        Compute area under Precision-Recall curve
        Return: float

    compute_confusion_matrix(y_true, y_pred)
        Return: 2x2 numpy array

    plot_pr_curve(y_true, y_prob, model_name)
        Plot precision-recall curve
        Save to results/plots/pr_curve_{model_name}.png

    plot_confusion_matrix(cm, model_name)
        Plot heatmap of confusion matrix
        Save to results/plots/cm_{model_name}.png

    compare_models(results_dict)
        Input: {"LogReg": metrics, "XGBoost": metrics, "GraphSAGE": metrics, "GAT": metrics}
        Print formatted comparison table
        Save to results/metrics.json

OUTPUT:
    Printed comparison table
    Saved plots in results/plots/
    results/metrics.json

TODO: Implement all functions below
"""

def compute_f1(y_true, y_pred):
    pass

def compute_pr_auc(y_true, y_prob):
    pass

def compute_confusion_matrix(y_true, y_pred):
    pass

def plot_pr_curve(y_true, y_prob, model_name):
    pass

def plot_confusion_matrix(cm, model_name):
    pass

def compare_models(results_dict):
    pass
