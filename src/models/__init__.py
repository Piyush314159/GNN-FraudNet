# Models package
from .gat import GAT
from .graphsage import GraphSAGE

# Baseline models require sklearn/xgboost — optional for inference-only setups
try:
    from .baseline import evaluate_baseline, predict_proba, train_logreg, train_xgboost
except ImportError:
    train_logreg = train_xgboost = predict_proba = evaluate_baseline = None

__all__ = [
    "GraphSAGE",
    "GAT",
    "train_logreg",
    "train_xgboost",
    "predict_proba",
    "evaluate_baseline",
]
