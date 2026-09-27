"""
test_prediction.py
==================
Tests for the prediction service.
"""
import pytest
import torch

from src.services.prediction import PredictionResult, PredictionService


class TestPredictionService:

    def test_predict_node_returns_result(self, prediction_service):
        result = prediction_service.predict_node(0)
        assert isinstance(result, PredictionResult)
        assert result.node_id == 0
        assert 0.0 <= result.fraud_probability <= 1.0
        assert result.label in ("illicit", "licit", "uncertain")
        assert result.confidence in ("high", "medium", "low")
        assert len(result.probabilities) == 2
        assert len(result.logits) == 2

    def test_predict_node_invalid_id(self, prediction_service):
        with pytest.raises(ValueError, match="not found"):
            prediction_service.predict_node(-1)
        with pytest.raises(ValueError, match="not found"):
            prediction_service.predict_node(999999)

    def test_predict_features_valid(self, prediction_service):
        features = [0.0] * 166
        result = prediction_service.predict_features(features)
        assert isinstance(result, PredictionResult)
        assert result.node_id is None
        assert 0.0 <= result.fraud_probability <= 1.0

    def test_predict_features_wrong_count(self, prediction_service):
        with pytest.raises(ValueError, match="Expected 166"):
            prediction_service.predict_features([0.0] * 100)

    def test_predict_features_empty(self, prediction_service):
        with pytest.raises(ValueError, match="Expected 166"):
            prediction_service.predict_features([])

    def test_probabilities_sum_to_one(self, prediction_service):
        result = prediction_service.predict_node(0)
        total = sum(result.probabilities)
        assert abs(total - 1.0) < 1e-5

    def test_service_not_loaded_raises(self):
        svc = PredictionService()
        with pytest.raises(RuntimeError, match="not loaded"):
            svc.predict_node(0)

    def test_ground_truth_label(self, prediction_service):
        # Node 0 is licit in our mock data (y[:10] = 0)
        label = prediction_service.get_ground_truth_label(0)
        assert label == "licit"

        # Node 10 is illicit in our mock data (y[10:15] = 1)
        label = prediction_service.get_ground_truth_label(10)
        assert label == "illicit"

        # Node 20 is unknown in our mock data
        label = prediction_service.get_ground_truth_label(20)
        assert label == "unknown"

    def test_multiple_predictions_consistent(self, prediction_service):
        """Same input should give same output (model is in eval mode)."""
        result1 = prediction_service.predict_node(5)
        result2 = prediction_service.predict_node(5)
        assert result1.fraud_probability == result2.fraud_probability
        assert result1.label == result2.label
