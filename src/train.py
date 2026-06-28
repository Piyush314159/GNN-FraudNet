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
    train_epoch(model, data, optimizer, loss_fn, class_weights)
        Set model to train mode
        Forward pass on ALL nodes
        Mask loss to only training nodes (data.train_mask)
        Backward pass + optimizer step
        Return: train loss (float)

    validate(model, data)
        Set model to eval mode
        Forward pass, mask to val nodes
        Compute val loss, val F1, val PR-AUC
        Return: dict of val metrics

    train(model, data, config)
        Loop for config['epochs']:
            Call train_epoch → get train loss
            Call validate → get val metrics
            Log metrics every 10 epochs
            Early stopping: if val F1 doesn't improve for config['patience'] epochs → stop
            Save best model checkpoint to results/best_model.pt
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

TODO: Implement all functions below
"""

def train_epoch(model, data, optimizer, loss_fn):
    # TODO: single training step with masking
    pass

def validate(model, data):
    # TODO: eval on val mask, return metrics dict
    pass

def train(model, data, config):
    # TODO: full loop with early stopping and checkpointing
    pass

def load_checkpoint(path, model):
    # TODO: load state dict, return model
    pass
