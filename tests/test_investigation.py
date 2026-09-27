"""
test_investigation.py
=====================
Tests for the investigation pipeline services.
"""
import pytest

from src.services.graph_analysis import GraphAnalysisResult, GraphAnalysisService
from src.services.investigation import InvestigationContext, InvestigationService
from src.services.prediction import PredictionResult


class TestGraphAnalysis:

    def test_analyze_node_valid(self, graph_analysis_service):
        result = graph_analysis_service.analyze_node(0)
        assert isinstance(result, GraphAnalysisResult)
        assert result.node_id == 0
        assert result.total_degree >= 0

    def test_analyze_node_invalid(self, graph_analysis_service):
        with pytest.raises(ValueError, match="not in graph"):
            graph_analysis_service.analyze_node(-1)

    def test_neighbor_counts_consistent(self, graph_analysis_service):
        result = graph_analysis_service.analyze_node(0)
        n = result.neighbors_1hop
        assert n.total == n.illicit + n.licit + n.unknown

    def test_ground_truth_label(self, graph_analysis_service):
        # Node 0 is licit in mock data
        result = graph_analysis_service.analyze_node(0)
        assert result.ground_truth_label == "licit"

        # Node 10 is illicit in mock data
        result = graph_analysis_service.analyze_node(10)
        assert result.ground_truth_label == "illicit"

        # Node 20 is unknown in mock data
        result = graph_analysis_service.analyze_node(20)
        assert result.ground_truth_label == "unknown"


class TestInvestigationService:

    def test_build_context_returns_context(self, investigation_service):
        context = investigation_service.build_context(0)
        assert isinstance(context, InvestigationContext)
        assert context.node_id == 0
        assert context.prediction is not None
        assert context.graph_analysis is not None
        assert context.duration_seconds >= 0

    def test_build_context_prediction_populated(self, investigation_service):
        context = investigation_service.build_context(0)
        p = context.prediction
        assert p is not None
        assert 0.0 <= p.fraud_probability <= 1.0
        assert p.label in ("illicit", "licit", "uncertain")

    def test_format_context_for_llm(self, investigation_service):
        context = investigation_service.build_context(0)
        sections = investigation_service.format_context_for_llm(context)
        assert "prediction_section" in sections
        assert "explanation_section" in sections
        assert "graph_section" in sections
        # All sections should be non-empty strings
        for key, value in sections.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_context_captures_errors_gracefully(self, investigation_service):
        """Build context for a valid node should have no errors."""
        context = investigation_service.build_context(0)
        assert len(context.errors) == 0


class TestSchemas:

    def test_fraud_response_valid(self):
        from api.schema import FraudResponse

        resp = FraudResponse(
            fraud_probability=0.85,
            label="illicit",
            confidence="high",
            node_id=42,
        )
        assert resp.fraud_probability == 0.85

    def test_fraud_response_invalid_probability(self):
        from api.schema import FraudResponse

        with pytest.raises(Exception):  # pydantic ValidationError
            FraudResponse(
                fraud_probability=1.5,  # out of range
                label="illicit",
                confidence="high",
            )

    def test_investigation_response_default(self):
        from api.schema import InvestigationResponse

        resp = InvestigationResponse(node_id=0)
        assert resp.node_id == 0
        assert resp.investigation_report == ""
        assert resp.retrieved_context == []
