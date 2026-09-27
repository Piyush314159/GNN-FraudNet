"""
investigation_agent.py
======================
LangGraph-based investigation workflow.

Implements a simple linear agent that orchestrates:
    predict → explain → analyze_graph → retrieve_knowledge → generate_report

Uses LangGraph's StateGraph for state management and step-by-step execution.
The workflow is deliberately kept linear — no unnecessary multi-agent complexity.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

from langgraph.graph import END, StateGraph

from src.llm.report_generator import ReportGenerator
from src.rag.retriever import RetrievedDocument, Retriever
from src.services.explainability import ExplanationResult, ExplainabilityService
from src.services.graph_analysis import GraphAnalysisResult, GraphAnalysisService
from src.services.investigation import InvestigationContext, InvestigationService
from src.services.prediction import PredictionResult, PredictionService

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────────────

class InvestigationState(dict):
    """
    State object passed through the LangGraph workflow.
    Accumulates results from each step.
    """
    pass


# ── Node Functions ───────────────────────────────────────────────────────────

def get_prediction(state: InvestigationState) -> dict:
    """Step 1: Get GNN fraud prediction for the target node."""
    prediction_svc: PredictionService = state["prediction_service"]
    node_id: int = state["node_id"]

    try:
        result = prediction_svc.predict_node(node_id)
        logger.info(
            "Step 1/5 [predict]: node=%d prob=%.4f label=%s",
            node_id, result.fraud_probability, result.label,
        )
        return {"prediction": result}
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return {"prediction": None, "errors": state.get("errors", []) + [f"Prediction: {e}"]}


def get_explanation(state: InvestigationState) -> dict:
    """Step 2: Run GNNExplainer (or SHAP) on the target node."""
    explain_svc: ExplainabilityService = state["explainability_service"]
    node_id: int = state["node_id"]
    include_shap: bool = state.get("include_shap", False)
    top_k: int = state.get("explanation_top_k", 10)

    try:
        if include_shap:
            result = explain_svc.explain_node_shap(node_id, top_k=top_k)
        else:
            result = explain_svc.explain_node_gnn(node_id, top_k=top_k)
        logger.info(
            "Step 2/5 [explain]: method=%s features=%d",
            result.method, len(result.top_features),
        )
        return {"explanation": result}
    except Exception as e:
        logger.error("Explanation failed: %s", e)
        return {"explanation": None, "errors": state.get("errors", []) + [f"Explanation: {e}"]}


def analyze_graph(state: InvestigationState) -> dict:
    """Step 3: Analyze the graph neighborhood of the target node."""
    graph_svc: GraphAnalysisService = state["graph_analysis_service"]
    node_id: int = state["node_id"]

    try:
        result = graph_svc.analyze_node(node_id)
        logger.info(
            "Step 3/5 [graph]: degree=%d illicit_neighbors=%d flags=%s",
            result.total_degree, result.neighbors_1hop.illicit,
            result.structural_flags,
        )
        return {"graph_analysis": result}
    except Exception as e:
        logger.error("Graph analysis failed: %s", e)
        return {"graph_analysis": None, "errors": state.get("errors", []) + [f"Graph analysis: {e}"]}


def retrieve_knowledge(state: InvestigationState) -> dict:
    """Step 4: Retrieve relevant fraud knowledge using RAG."""
    retriever: Optional[Retriever] = state.get("retriever")

    if retriever is None:
        logger.warning("Step 4/5 [RAG]: No retriever available, skipping")
        return {"retrieved_docs": []}

    # Build query from accumulated context
    prediction: Optional[PredictionResult] = state.get("prediction")
    explanation: Optional[ExplanationResult] = state.get("explanation")
    graph_analysis: Optional[GraphAnalysisResult] = state.get("graph_analysis")

    top_features = []
    if explanation and explanation.top_features:
        top_features = [f.feature_name for f in explanation.top_features[:5]]

    structural_flags = []
    if graph_analysis and graph_analysis.structural_flags:
        structural_flags = graph_analysis.structural_flags

    neighbor_summary = ""
    if graph_analysis:
        n = graph_analysis.neighbors_1hop
        neighbor_summary = (
            f"{n.total} neighbors, {n.illicit} illicit, {n.licit} licit, "
            f"{n.unknown} unknown"
        )

    query = retriever.build_query_from_context(
        prediction_label=prediction.label if prediction else "unknown",
        top_features=top_features,
        structural_flags=structural_flags,
        neighbor_summary=neighbor_summary,
    )

    try:
        docs = retriever.retrieve(query)
        logger.info("Step 4/5 [RAG]: Retrieved %d chunks", len(docs))
        return {"retrieved_docs": docs}
    except Exception as e:
        logger.error("RAG retrieval failed: %s", e)
        return {"retrieved_docs": [], "errors": state.get("errors", []) + [f"RAG: {e}"]}


def generate_report(state: InvestigationState) -> dict:
    """Step 5: Generate the investigation report using the LLM."""
    report_gen: Optional[ReportGenerator] = state.get("report_generator")
    investigation_svc: InvestigationService = state["investigation_service"]

    # Build InvestigationContext from accumulated state
    node_id = state["node_id"]
    context = InvestigationContext(
        node_id=node_id,
        prediction=state.get("prediction"),
        explanation=state.get("explanation"),
        graph_analysis=state.get("graph_analysis"),
        retrieved_context=[],
        errors=state.get("errors", []),
    )

    retrieved_docs = state.get("retrieved_docs", [])

    if report_gen is None:
        logger.warning("Step 5/5 [LLM]: No report generator, using fallback")
        sections = investigation_svc.format_context_for_llm(context)
        report = ReportGenerator._fallback_report(context, sections)
    else:
        try:
            report = report_gen.generate_report(
                context, retrieved_docs, investigation_svc
            )
            logger.info("Step 5/5 [LLM]: Report generated (%d chars)", len(report))
        except Exception as e:
            logger.error("LLM report generation failed: %s", e)
            sections = investigation_svc.format_context_for_llm(context)
            report = ReportGenerator._fallback_report(context, sections)
            state.setdefault("errors", []).append(f"LLM: {e}")

    return {"investigation_report": report}


# ── Build the Graph ──────────────────────────────────────────────────────────

def build_investigation_graph() -> StateGraph:
    """
    Construct the LangGraph investigation workflow.

    Returns a compiled StateGraph that can be invoked with:
        result = graph.invoke(initial_state)
    """
    workflow = StateGraph(InvestigationState)

    # Add nodes
    workflow.add_node("get_prediction", get_prediction)
    workflow.add_node("get_explanation", get_explanation)
    workflow.add_node("analyze_graph", analyze_graph)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("generate_report", generate_report)

    # Linear edges: predict → explain → graph → RAG → report
    workflow.set_entry_point("get_prediction")
    workflow.add_edge("get_prediction", "get_explanation")
    workflow.add_edge("get_explanation", "analyze_graph")
    workflow.add_edge("analyze_graph", "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


# ── High-Level API ───────────────────────────────────────────────────────────

class InvestigationAgent:
    """
    High-level interface for running fraud investigations.

    Wraps the LangGraph workflow with service injection.
    """

    def __init__(
        self,
        prediction_service: PredictionService,
        explainability_service: ExplainabilityService,
        graph_analysis_service: GraphAnalysisService,
        investigation_service: InvestigationService,
        retriever: Optional[Retriever] = None,
        report_generator: Optional[ReportGenerator] = None,
    ):
        self.prediction_service = prediction_service
        self.explainability_service = explainability_service
        self.graph_analysis_service = graph_analysis_service
        self.investigation_service = investigation_service
        self.retriever = retriever
        self.report_generator = report_generator
        self._graph = build_investigation_graph()

    def investigate(
        self,
        node_id: int,
        include_shap: bool = False,
        explanation_top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Run a full investigation on a node.

        Args:
            node_id: The graph node to investigate.
            include_shap: Run SHAP instead of GNNExplainer (slower).
            explanation_top_k: Number of top features to include.

        Returns:
            Complete investigation results as a dictionary.
        """
        start_time = time.time()

        initial_state = InvestigationState({
            "node_id": node_id,
            "include_shap": include_shap,
            "explanation_top_k": explanation_top_k,
            # Inject services
            "prediction_service": self.prediction_service,
            "explainability_service": self.explainability_service,
            "graph_analysis_service": self.graph_analysis_service,
            "investigation_service": self.investigation_service,
            "retriever": self.retriever,
            "report_generator": self.report_generator,
            # Accumulated results
            "errors": [],
        })

        result = self._graph.invoke(initial_state)
        duration = time.time() - start_time

        logger.info(
            "Investigation complete for node %d in %.2fs", node_id, duration
        )

        return {
            "node_id": node_id,
            "prediction": result.get("prediction"),
            "explanation": result.get("explanation"),
            "graph_analysis": result.get("graph_analysis"),
            "retrieved_docs": result.get("retrieved_docs", []),
            "investigation_report": result.get("investigation_report", ""),
            "errors": result.get("errors", []),
            "duration_seconds": round(duration, 3),
        }
