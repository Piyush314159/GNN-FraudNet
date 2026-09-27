"""
schema.py
=========
Pydantic request/response models for all API endpoints.

Extended from the original 2-model schema to include investigation responses.
"""
from typing import Optional

from pydantic import BaseModel, Field, create_model


# ── Existing Schemas (preserved) ─────────────────────────────────────────────

# Dynamically build TransactionRequest with feature_1 .. feature_166
_fields = {f"feature_{i}": (float, Field(...)) for i in range(1, 167)}
TransactionRequest = create_model("TransactionRequest", **_fields)


class FraudResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    label: str  # "illicit" or "licit"
    confidence: str  # "high", "medium", or "low"
    node_id: Optional[int] = None


# ── New Schemas ──────────────────────────────────────────────────────────────

class FeatureImportanceSchema(BaseModel):
    feature_name: str
    importance: float
    direction: str
    rank: int


class PredictionDetail(BaseModel):
    node_id: Optional[int] = None
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    label: str
    confidence: str
    probabilities: list[float] = []


class ExplanationDetail(BaseModel):
    method: str
    top_features: list[FeatureImportanceSchema] = []
    error: Optional[str] = None


class NeighborSummarySchema(BaseModel):
    total: int = 0
    illicit: int = 0
    licit: int = 0
    unknown: int = 0
    illicit_ratio: float = 0.0


class GraphAnalysisDetail(BaseModel):
    node_id: int
    in_degree: int = 0
    out_degree: int = 0
    total_degree: int = 0
    ground_truth_label: str = "unknown"
    neighbors_1hop: NeighborSummarySchema = NeighborSummarySchema()
    neighbors_2hop: NeighborSummarySchema = NeighborSummarySchema()
    structural_flags: list[str] = []


class RetrievedDocumentSchema(BaseModel):
    text: str
    source: str
    relevance_score: float
    chunk_id: str


class InvestigationMetadata(BaseModel):
    duration_seconds: float = 0.0
    model_name: str = "GraphSAGE"
    explanation_method: str = "gnn_explainer"
    errors: list[str] = []


class InvestigationResponse(BaseModel):
    """Complete response from the /investigate endpoint."""

    node_id: int
    prediction: Optional[PredictionDetail] = None
    explanation: Optional[ExplanationDetail] = None
    graph_analysis: Optional[GraphAnalysisDetail] = None
    retrieved_context: list[RetrievedDocumentSchema] = []
    investigation_report: str = ""
    metadata: InvestigationMetadata = InvestigationMetadata()


class InvestigationRequest(BaseModel):
    """Optional parameters for the investigation endpoint."""

    include_shap: bool = Field(
        default=False,
        description="Use SHAP instead of GNNExplainer (slower but more detailed)",
    )
    explanation_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top features to include in explanation",
    )


class BatchPredictResponse(BaseModel):
    """Response for batch prediction endpoint."""

    predictions: list[FraudResponse]
    total: int
    summary: dict
