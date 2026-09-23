"""
train.py
========
PURPOSE:
    Full training loop for GNN models (GraphSAGE and GAT).
    Handles epochs, loss computation, validation, early stopping,
    and model checkpointing.

INPUTS:
    - model     : GraphSAGE or GAT instance
    - data      : PyG Data object with masks
    - config    : dict with hyperparameters (lr, epochs, patience, etc.)

HOW IT WORKS:
    train_epoch(model, data, optimizer, loss_fn)
        Set model to train mode
        Forward pass on ALL nodes
        Mask loss to only training nodes (data.train_mask)
        Backward pass + optimizer step
        Return: train loss (float)

    validate(model, data, loss_fn)
        Set model to eval mode
        Forward pass, mask to val nodes
        Compute val loss, val F1, val PR-AUC
        Return: dict of val metrics

    train(model, data, config, class_weights, model_name)
        Loop for config['epochs']:
            Call train_epoch → get train loss
            Call validate → get val metrics
            Log metrics every 10 epochs
            Early stopping: if val F1 doesn't improve for config['patience'] epochs → stop
            Save best model checkpoint to results/best_{model_name}.pt
        Return: trained model, history dict (loss/metrics per epoch)

    load_checkpoint(path, model)
        Load saved state dict into model
        Return: model with loaded weights

CONFIG KEYS:
    lr        : learning rate (suggest 0.001)
    epochs    : max epochs (suggest 200)
    patience  : early stopping patience (suggest 20)
    dropout   : dropout rate (suggest 0.5)
    weight_decay : L2 regularization (suggest 5e-4)

OUTPUT:
    Trained model
    History dict for plotting loss curves
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score


@torch.no_grad()
def _compute_metrics(logits, labels, mask):
    logits_masked = logits[mask]
    labels_masked = labels[mask]

    probs = F.softmax(logits_masked, dim=1)
    preds = logits_masked.argmax(dim=1)

    y_true = labels_masked.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_prob = probs[:, 1].cpu().numpy()

    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob, pos_label=1) if len(np.unique(y_true)) > 1 else 0.0

    return {"f1": f1, "pr_auc": pr_auc, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def train_epoch(model, data, optimizer, loss_fn):
    """Single training step with masking. Returns the train loss (float)."""
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate(model, data, loss_fn):
    """Eval on val_mask, return a dict of loss/F1/PR-AUC metrics."""
    model.eval()
    logits = model(data.x, data.edge_index)

    metrics = _compute_metrics(logits, data.y, data.val_mask)
    metrics["val_loss"] = loss_fn(logits[data.val_mask], data.y[data.val_mask]).item()

    return metrics


def train(model, data, config, class_weights=None, model_name="model", checkpoint_dir="results"):
    """Full training loop with early stopping and best-checkpoint saving."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.001),
        weight_decay=config.get("weight_decay", 5e-4),
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    epochs = config.get("epochs", 200)
    patience = config.get("patience", 20)

    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    checkpoint_path = os.path.join(checkpoint_dir, f"best_{safe_name}.pt")

    best_val_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_pr_auc": []}

    print(f"\n{'=' * 60}")
    print(f"  Training {model_name}")
    print(f"{'=' * 60}")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, data, optimizer, loss_fn)
        val_metrics = validate(model, data, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d}/{epochs} — "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val PR-AUC: {val_metrics['pr_auc']:.4f}"
            )

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    model = load_checkpoint(checkpoint_path, model)
    print(f"  Best val F1: {best_val_f1:.4f}")

    return model, history


def load_checkpoint(path, model):
    """Load a saved state dict into model, return the model."""
    model.load_state_dict(torch.load(path, weights_only=True))
    return model
