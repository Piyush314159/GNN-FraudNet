"""
test_api.py
===========
Integration tests for FastAPI endpoints.

Uses TestClient with a mock application that has synthetic data
to avoid loading the real graph.
"""
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def test_client(mock_model, mock_graph_data):
    """Create a test client with mock services."""
    from src.services.prediction import PredictionService
    from src.services.explainability import ExplainabilityService
    from src.services.graph_analysis import GraphAnalysisService
    from src.services.investigation import InvestigationService

    # Import app without triggering lifespan (we'll set up state manually)
    from api.main import app

    # Manually set up app state
    pred_svc = PredictionService()
    pred_svc.model = mock_model
    pred_svc.data = mock_graph_data

    explain_svc = ExplainabilityService(mock_model, mock_graph_data)
    graph_svc = GraphAnalysisService(mock_graph_data)
    investigation_svc = InvestigationService(pred_svc, explain_svc, graph_svc)

    app.state.model = mock_model
    app.state.data = mock_graph_data
    app.state.prediction_service = pred_svc
    app.state.explainability_service = explain_svc
    app.state.graph_analysis_service = graph_svc
    app.state.investigation_service = investigation_svc
    app.state.retriever = None
    app.state.report_generator = None
    app.state.investigation_agent = None  # Skip agent for basic tests

    # Create client without lifespan to avoid loading real model
    client = TestClient(app, raise_server_exceptions=False)
    return client


class TestHealthEndpoint:

    def test_health_returns_ok(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"] == "GraphSAGE"


class TestPredictEndpoint:

    def test_predict_valid_features(self, test_client):
        payload = {f"feature_{i}": 0.0 for i in range(1, 167)}
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "label" in data
        assert "confidence" in data
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_missing_features(self, test_client):
        payload = {"feature_1": 1.0}  # missing features 2-166
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 422  # validation error

    def test_predict_empty_body(self, test_client):
        response = test_client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_by_node_valid(self, test_client):
        response = test_client.get("/predict/0")
        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == 0
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_by_node_not_found(self, test_client):
        response = test_client.get("/predict/999999")
        assert response.status_code == 404


class TestNodeEndpoint:

    def test_node_info_valid(self, test_client):
        response = test_client.get("/node/0")
        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == 0
        assert "ground_truth_label" in data
        assert "in_degree" in data
        assert "out_degree" in data

    def test_node_info_not_found(self, test_client):
        response = test_client.get("/node/999999")
        assert response.status_code == 404


class TestDashboard:

    def test_dashboard_returns_html(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
