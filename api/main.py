"""
main.py
=======
PURPOSE:
    FastAPI application exposing the trained GNN model as a REST API.
    Allows any system to send transaction features and get a fraud score back.

ENDPOINTS:
    GET  /                    → Interactive fraud detection dashboard
    GET  /health              → returns {"status": "ok", "model": "GraphSAGE", ...}
    POST /predict             → takes TransactionRequest, returns FraudResponse
    GET  /predict/{node_id}   → predict for a known node ID in the Elliptic graph
    POST /investigate/{node_id} → full investigation with explanation, RAG, LLM report

HOW TO RUN:
    uvicorn api.main:app --reload --port 8000
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.schema import (
    BatchPredictResponse,
    ExplanationDetail,
    FeatureImportanceSchema,
    FraudResponse,
    GraphAnalysisDetail,
    InvestigationMetadata,
    InvestigationRequest,
    InvestigationResponse,
    NeighborSummarySchema,
    PredictionDetail,
    RetrievedDocumentSchema,
    TransactionRequest,
)
from src.config import (
    APP_HOST,
    APP_PORT,
    DATA_PATH,
    GEMINI_API_KEY,
    LOG_LEVEL,
    MODEL_PATH,
    NUM_FEATURES,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _confidence_bucket(prob):
    margin = abs(prob - 0.5)
    if margin >= 0.3:
        return "high"
    if margin >= 0.1:
        return "medium"
    return "low"


def _init_services(app: FastAPI) -> None:
    """Initialize all services and store them in app.state."""
    from src.services.prediction import PredictionService
    from src.services.explainability import ExplainabilityService
    from src.services.graph_analysis import GraphAnalysisService
    from src.services.investigation import InvestigationService

    # Prediction service
    pred_svc = PredictionService(model_path=MODEL_PATH, data_path=DATA_PATH)
    pred_svc.load()
    app.state.prediction_service = pred_svc

    # Keep backward compat for existing endpoints
    app.state.model = pred_svc.model
    app.state.data = pred_svc.data

    # Explainability service
    explain_svc = ExplainabilityService(pred_svc.model, pred_svc.data)
    app.state.explainability_service = explain_svc

    # Graph analysis service
    graph_svc = GraphAnalysisService(pred_svc.data)
    app.state.graph_analysis_service = graph_svc

    # Investigation service
    investigation_svc = InvestigationService(pred_svc, explain_svc, graph_svc)
    app.state.investigation_service = investigation_svc

    # RAG (optional — may fail if dependencies not installed)
    retriever = None
    try:
        from src.rag.knowledge_base import KnowledgeBase
        from src.rag.retriever import Retriever

        kb = KnowledgeBase()
        kb.initialize()
        retriever = Retriever(kb)
        app.state.retriever = retriever
        logger.info("RAG knowledge base initialized")
    except Exception as e:
        logger.warning("RAG initialization failed (non-critical): %s", e)
        app.state.retriever = None

    # LLM report generator (optional — requires API key)
    report_gen = None
    try:
        from src.llm.report_generator import ReportGenerator

        if GEMINI_API_KEY:
            report_gen = ReportGenerator()
            logger.info("LLM report generator initialized (model: %s)", report_gen.model)
        else:
            logger.warning("GEMINI_API_KEY not set — LLM reports will use fallback")
    except Exception as e:
        logger.warning("LLM initialization failed (non-critical): %s", e)
    app.state.report_generator = report_gen

    # Investigation agent
    try:
        from src.agents.investigation_agent import InvestigationAgent

        agent = InvestigationAgent(
            prediction_service=pred_svc,
            explainability_service=explain_svc,
            graph_analysis_service=graph_svc,
            investigation_service=investigation_svc,
            retriever=retriever,
            report_generator=report_gen,
        )
        app.state.investigation_agent = agent
        logger.info("Investigation agent initialized")
    except Exception as e:
        logger.warning("Investigation agent initialization failed: %s", e)
        app.state.investigation_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_services(app)
    yield


app = FastAPI(
    title="GNN-FraudNet",
    description="AI-Powered Fraud Detection & Investigation Platform",
    version="2.0.0",
    lifespan=lifespan,
)

# Serve static files (CSS, JS, images if any)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Existing Endpoints (preserved) ──────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the interactive fraud detection dashboard."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "GraphSAGE",
        "rag_available": app.state.retriever is not None,
        "llm_available": app.state.report_generator is not None,
        "investigation_available": app.state.investigation_agent is not None,
    }


@app.post("/predict", response_model=FraudResponse)
def predict(request: TransactionRequest):
    features = [getattr(request, f"feature_{i}") for i in range(1, NUM_FEATURES + 1)]
    x = torch.tensor([features], dtype=torch.float)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    with torch.no_grad():
        logits = app.state.model(x, edge_index)
        probs = torch.softmax(logits, dim=1)[0]

    fraud_prob = probs[1].item()
    return FraudResponse(
        fraud_probability=fraud_prob,
        label="illicit" if fraud_prob >= 0.5 else "licit",
        confidence=_confidence_bucket(fraud_prob),
    )


@app.get("/predict/{node_id}", response_model=FraudResponse)
def predict_by_node(node_id: int):
    data = app.state.data
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    with torch.no_grad():
        logits = app.state.model(data.x, data.edge_index)
        probs = torch.softmax(logits[node_id], dim=0)

    fraud_prob = probs[1].item()
    return FraudResponse(
        fraud_probability=fraud_prob,
        label="illicit" if fraud_prob >= 0.5 else "licit",
        confidence=_confidence_bucket(fraud_prob),
        node_id=node_id,
    )


# ── New Endpoints ────────────────────────────────────────────────────────────


@app.post("/investigate/{node_id}", response_model=InvestigationResponse)
def investigate_node(
    node_id: int,
    request: Optional[InvestigationRequest] = None,
):
    """
    Run a full fraud investigation on a graph node.

    Orchestrates: GNN prediction → explainability → graph analysis →
    RAG retrieval → LLM report generation.
    """
    agent = app.state.investigation_agent
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Investigation agent not available. Check server logs.",
        )

    data = app.state.data
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    params = request or InvestigationRequest()

    try:
        result = agent.investigate(
            node_id=node_id,
            include_shap=params.include_shap,
            explanation_top_k=params.explanation_top_k,
        )
    except Exception as e:
        logger.error("Investigation failed for node %d: %s", node_id, e)
        raise HTTPException(status_code=500, detail=f"Investigation failed: {e}")

    return _format_investigation_response(result)


@app.get("/investigate/stream/{node_id}")
def investigate_node_stream(
    node_id: int,
    include_shap: bool = False,
    explanation_top_k: int = 10,
):
    """
    Stream investigation results (SSE).
    Orchestrates: GNN prediction → explainability → graph analysis →
    RAG retrieval → LLM report generation (streamed).
    """
    agent = app.state.investigation_agent
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Investigation agent not available. Check server logs.",
        )

    data = app.state.data
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    return StreamingResponse(
        agent.investigate_stream(
            node_id=node_id,
            include_shap=include_shap,
            explanation_top_k=explanation_top_k,
        ),
        media_type="text/event-stream"
    )


@app.get("/node/{node_id}")
def get_node_info(node_id: int):
    """Get basic information about a node (label, degree, neighbors)."""
    graph_svc = app.state.graph_analysis_service
    data = app.state.data

    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    try:
        analysis = graph_svc.analyze_node(node_id)
        return {
            "node_id": node_id,
            "ground_truth_label": analysis.ground_truth_label,
            "in_degree": analysis.in_degree,
            "out_degree": analysis.out_degree,
            "total_degree": analysis.total_degree,
            "neighbors_1hop": {
                "total": analysis.neighbors_1hop.total,
                "illicit": analysis.neighbors_1hop.illicit,
                "licit": analysis.neighbors_1hop.licit,
                "unknown": analysis.neighbors_1hop.unknown,
            },
            "structural_flags": analysis.structural_flags,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Response Formatting ─────────────────────────────────────────────────────


def _format_investigation_response(result: dict) -> InvestigationResponse:
    """Convert raw investigation result dict to Pydantic response model."""
    # Prediction
    prediction = None
    if result.get("prediction"):
        p = result["prediction"]
        prediction = PredictionDetail(
            node_id=p.node_id,
            fraud_probability=p.fraud_probability,
            label=p.label,
            confidence=p.confidence,
            probabilities=p.probabilities,
        )

    # Explanation
    explanation = None
    if result.get("explanation"):
        e = result["explanation"]
        explanation = ExplanationDetail(
            method=e.method,
            top_features=[
                FeatureImportanceSchema(
                    feature_name=f.feature_name,
                    importance=f.importance,
                    direction=f.direction,
                    rank=f.rank,
                )
                for f in (e.top_features or [])
            ],
            error=e.error,
        )

    # Graph analysis
    graph_analysis = None
    if result.get("graph_analysis"):
        g = result["graph_analysis"]
        graph_analysis = GraphAnalysisDetail(
            node_id=g.node_id,
            in_degree=g.in_degree,
            out_degree=g.out_degree,
            total_degree=g.total_degree,
            ground_truth_label=g.ground_truth_label,
            neighbors_1hop=NeighborSummarySchema(
                total=g.neighbors_1hop.total,
                illicit=g.neighbors_1hop.illicit,
                licit=g.neighbors_1hop.licit,
                unknown=g.neighbors_1hop.unknown,
                illicit_ratio=g.neighbors_1hop.illicit_ratio,
            ),
            neighbors_2hop=NeighborSummarySchema(
                total=g.neighbors_2hop.total,
                illicit=g.neighbors_2hop.illicit,
                licit=g.neighbors_2hop.licit,
                unknown=g.neighbors_2hop.unknown,
                illicit_ratio=g.neighbors_2hop.illicit_ratio,
            ),
            structural_flags=g.structural_flags,
        )

    # Retrieved context
    retrieved_context = []
    for doc in result.get("retrieved_docs", []):
        retrieved_context.append(
            RetrievedDocumentSchema(
                text=doc.text,
                source=doc.source,
                relevance_score=doc.relevance_score,
                chunk_id=doc.chunk_id,
            )
        )

    # Metadata
    explanation_method = "gnn_explainer"
    if result.get("explanation"):
        explanation_method = result["explanation"].method

    metadata = InvestigationMetadata(
        duration_seconds=result.get("duration_seconds", 0.0),
        model_name="GraphSAGE",
        explanation_method=explanation_method,
        errors=result.get("errors", []),
    )

    return InvestigationResponse(
        node_id=result["node_id"],
        prediction=prediction,
        explanation=explanation,
        graph_analysis=graph_analysis,
        retrieved_context=retrieved_context,
        investigation_report=result.get("investigation_report", ""),
        metadata=metadata,
    )
