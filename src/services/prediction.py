"""
prediction.py
=============
Prediction service — wraps GNN model inference into a clean, reusable API.

Extracts the inference logic from api/main.py so it can be used by both
the API layer and the investigation pipeline without duplication.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.config import (
    DATA_PATH,
    FRAUD_THRESHOLD_HIGH,
    FRAUD_THRESHOLD_LOW,
    HIDDEN_CHANNELS,
    MODEL_PATH,
    NUM_CLASSES,
    NUM_FEATURES,
)
from src.models.graphsage import GraphSAGE

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured output from model inference."""

    node_id: Optional[int]
    fraud_probability: float
    label: str                  # "illicit", "licit", or "uncertain"
    confidence: str             # "high", "medium", or "low"
    logits: list[float]         # raw model output [licit_logit, illicit_logit]
    probabilities: list[float]  # softmax output [licit_prob, illicit_prob]


def _classify(prob: float) -> str:
    """Map fraud probability to a label."""
    if prob >= FRAUD_THRESHOLD_HIGH:
        return "illicit"
    elif prob <= FRAUD_THRESHOLD_LOW:
        return "licit"
    return "uncertain"


def _confidence_bucket(prob: float) -> str:
    """Map fraud probability to a confidence level."""
    margin = abs(prob - 0.5)
    if margin >= 0.3:
        return "high"
    if margin >= 0.1:
        return "medium"
    return "low"


class PredictionService:
    """Manages model lifecycle and provides inference methods."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        data_path: str = DATA_PATH,
    ):
        self.model: Optional[GraphSAGE] = None
        self.data = None
        self._model_path = model_path
        self._data_path = data_path

    def load(self) -> None:
        """Load model weights and graph data from disk."""
        logger.info("Loading GraphSAGE model from %s", self._model_path)
        self.model = GraphSAGE(
            in_channels=NUM_FEATURES,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=NUM_CLASSES,
        )
        self.model.load_state_dict(
            torch.load(self._model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()

        logger.info("Loading graph data from %s", self._data_path)
        self.data = torch.load(self._data_path, map_location="cpu", weights_only=False)
        logger.info(
            "Loaded graph: %d nodes, %d edges",
            self.data.num_nodes,
            self.data.edge_index.size(1),
        )

    def _ensure_loaded(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("PredictionService not loaded. Call .load() first.")

    def predict_node(self, node_id: int) -> PredictionResult:
        """Predict fraud probability for a known node using the full graph."""
        self._ensure_loaded()
        if node_id < 0 or node_id >= self.data.num_nodes:
            raise ValueError(f"Node {node_id} not found (valid: 0–{self.data.num_nodes - 1})")

        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(logits[node_id], dim=0)

        fraud_prob = probs[1].item()
        return PredictionResult(
            node_id=node_id,
            fraud_probability=fraud_prob,
            label=_classify(fraud_prob),
            confidence=_confidence_bucket(fraud_prob),
            logits=logits[node_id].tolist(),
            probabilities=probs.tolist(),
        )

    def predict_features(self, features: list[float]) -> PredictionResult:
        """Predict fraud probability from raw 166-dimensional feature vector."""
        self._ensure_loaded()
        if len(features) != NUM_FEATURES:
            raise ValueError(f"Expected {NUM_FEATURES} features, got {len(features)}")

        x = torch.tensor([features], dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        with torch.no_grad():
            logits = self.model(x, edge_index)
            probs = F.softmax(logits[0], dim=0)

        fraud_prob = probs[1].item()
        return PredictionResult(
            node_id=None,
            fraud_probability=fraud_prob,
            label=_classify(fraud_prob),
            confidence=_confidence_bucket(fraud_prob),
            logits=logits[0].tolist(),
            probabilities=probs.tolist(),
        )

    def get_ground_truth_label(self, node_id: int) -> Optional[str]:
        """Return the dataset's ground truth label if available."""
        self._ensure_loaded()
        label = self.data.y[node_id].item()
        if label == 1:
            return "illicit"
        elif label == 0:
            return "licit"
        return "unknown"
