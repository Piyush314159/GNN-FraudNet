"""
conftest.py
===========
Shared pytest fixtures for GNN-FraudNet tests.

Provides mock model, data, and service fixtures that avoid loading
the real 141MB graph data during testing.
"""
import os
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_graph_data():
    """Create a small synthetic graph for testing."""
    num_nodes = 50
    num_features = 166
    num_edges = 100

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.full((num_nodes,), -1, dtype=torch.long)

    # Label some nodes
    y[:10] = 0   # licit
    y[10:15] = 1  # illicit
    # rest are unknown (-1)

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:10] = True
    val_mask[10:12] = True
    test_mask[12:15] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data


@pytest.fixture
def mock_model():
    """Create a fresh (untrained) GraphSAGE model for testing."""
    from src.models.graphsage import GraphSAGE

    model = GraphSAGE(in_channels=166, hidden_channels=32, out_channels=2)
    model.eval()
    return model


@pytest.fixture
def prediction_service(mock_model, mock_graph_data):
    """Create a prediction service with mock model and data."""
    from src.services.prediction import PredictionService

    svc = PredictionService()
    svc.model = mock_model
    svc.data = mock_graph_data
    return svc


@pytest.fixture
def explainability_service(mock_model, mock_graph_data):
    """Create an explainability service with mock model and data."""
    from src.services.explainability import ExplainabilityService

    return ExplainabilityService(mock_model, mock_graph_data)


@pytest.fixture
def graph_analysis_service(mock_graph_data):
    """Create a graph analysis service with mock data."""
    from src.services.graph_analysis import GraphAnalysisService

    return GraphAnalysisService(mock_graph_data)


@pytest.fixture
def investigation_service(prediction_service, explainability_service, graph_analysis_service):
    """Create an investigation service with mock services."""
    from src.services.investigation import InvestigationService

    return InvestigationService(
        prediction_service, explainability_service, graph_analysis_service
    )
