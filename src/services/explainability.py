"""
explainability.py
=================
Explainability service — wraps SHAP/GNNExplainer into API-callable functions
that return structured dictionaries instead of raw tensors.

Imports and calls functions from the existing src/explain.py without modifying it.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from src.config import NUM_FEATURES

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """A single feature's importance from SHAP or GNNExplainer."""

    feature_name: str
    importance: float
    direction: str  # "increases_fraud", "decreases_fraud", or "neutral"
    rank: int


@dataclass
class ExplanationResult:
    """Structured output from explainability analysis."""

    node_id: int
    method: str  # "gnn_explainer" or "shap"
    top_features: list[FeatureImportance] = field(default_factory=list)
    edge_importances: Optional[list[float]] = None
    raw_feature_mask: Optional[list[float]] = None
    error: Optional[str] = None


class ExplainabilityService:
    """Provides structured explainability for GNN predictions."""

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def explain_node_gnn(
        self, node_id: int, top_k: int = 10, epochs: int = 200
    ) -> ExplanationResult:
        """
        Run GNNExplainer on a single node and return structured results.

        Uses src.explain.run_gnn_explainer() internally.
        """
        try:
            from src.explain import run_gnn_explainer

            node_mask, edge_mask = run_gnn_explainer(
                self.model, self.data, node_id, epochs=epochs
            )

            # node_mask shape: (num_nodes, num_features) or (1, num_features)
            # We want the feature importances for the target node
            if node_mask.dim() == 2:
                feature_importance = node_mask.mean(dim=0).detach().cpu().numpy()
            else:
                feature_importance = node_mask.detach().cpu().numpy()

            top_features = self._rank_features(feature_importance, top_k)

            return ExplanationResult(
                node_id=node_id,
                method="gnn_explainer",
                top_features=top_features,
                edge_importances=edge_mask.detach().cpu().tolist() if edge_mask is not None else None,
                raw_feature_mask=feature_importance.tolist(),
            )
        except Exception as e:
            logger.error("GNNExplainer failed for node %d: %s", node_id, e)
            return ExplanationResult(
                node_id=node_id,
                method="gnn_explainer",
                error=str(e),
            )

    def explain_node_shap(
        self, node_id: int, top_k: int = 10, nsamples: int = 50
    ) -> ExplanationResult:
        """
        Run SHAP KernelExplainer on a single node and return structured results.

        Uses src.explain.run_shap() internally.
        Note: SHAP is computationally expensive (~5-15 seconds per node).
        """
        try:
            from src.explain import run_shap

            shap_values = run_shap(
                self.model,
                self.data,
                node_indices=[node_id],
                n_explain=1,
                n_background=30,
                nsamples=nsamples,
            )

            feature_importance = shap_values[0]  # shape: (num_features,)
            top_features = self._rank_features(
                feature_importance, top_k, use_signed=True
            )

            return ExplanationResult(
                node_id=node_id,
                method="shap",
                top_features=top_features,
                raw_feature_mask=feature_importance.tolist(),
            )
        except Exception as e:
            logger.error("SHAP failed for node %d: %s", node_id, e)
            return ExplanationResult(
                node_id=node_id,
                method="shap",
                error=str(e),
            )

    @staticmethod
    def _rank_features(
        importances: np.ndarray, top_k: int, use_signed: bool = False
    ) -> list[FeatureImportance]:
        """Convert raw importance array to ranked FeatureImportance list."""
        abs_imp = np.abs(importances)
        top_indices = np.argsort(abs_imp)[-top_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            value = float(importances[idx])
            if use_signed:
                if value > 0.001:
                    direction = "increases_fraud"
                elif value < -0.001:
                    direction = "decreases_fraud"
                else:
                    direction = "neutral"
            else:
                direction = "increases_fraud" if value > 0 else "neutral"

            results.append(
                FeatureImportance(
                    feature_name=f"feature_{idx + 1}",
                    importance=abs(value),
                    direction=direction,
                    rank=rank,
                )
            )
        return results
