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
"""
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)

PLOT_DIR = "results/plots"
METRICS_PATH = "results/metrics.json"


def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1)


def compute_pr_auc(y_true, y_prob):
    return average_precision_score(y_true, y_prob, pos_label=1)


def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def plot_pr_curve(y_true, y_prob, model_name):
    os.makedirs(PLOT_DIR, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    pr_auc = compute_pr_auc(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#e74c3c", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.2, color="#e74c3c")
    ax.set_title(f"{model_name} — Precision-Recall Curve (AUC={pr_auc:.3f})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    save_path = os.path.join(PLOT_DIR, f"pr_curve_{safe_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion_matrix(cm, model_name):
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds", ax=ax,
        xticklabels=["Licit", "Illicit"], yticklabels=["Licit", "Illicit"],
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    save_path = os.path.join(PLOT_DIR, f"cm_{safe_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def compare_models(results_dict):
    """
    results_dict: {"LogReg": {"f1": ..., "pr_auc": ...}, "XGBoost": {...}, ...}
    Prints a formatted comparison table and saves results/metrics.json.
    """
    print("=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<20s} {'F1 (Illicit)':>14s} {'PR-AUC':>10s}")
    print("-" * 60)
    for name, metrics in results_dict.items():
        print(f"  {name:<20s} {metrics['f1']:>14.4f} {metrics['pr_auc']:>10.4f}")
    print("=" * 60)

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    serializable = {
        name: {"f1": metrics["f1"], "pr_auc": metrics["pr_auc"]}
        for name, metrics in results_dict.items()
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)

    return serializable
