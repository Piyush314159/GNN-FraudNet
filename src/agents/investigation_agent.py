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
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.llm.report_generator import ReportGenerator
from src.rag.retriever import RetrievedDocument, Retriever
from src.services.explainability import ExplanationResult, ExplainabilityService
from src.services.graph_analysis import GraphAnalysisResult, GraphAnalysisService
from src.services.investigation import InvestigationContext, InvestigationService
from src.services.prediction import PredictionResult, PredictionService

logger = logging.getLogger(__name__)


# ── State Schema ─────────────────────────────────────────────────────────────

class InvestigationState(TypedDict, total=False):
    """State object passed through the LangGraph workflow."""

    node_id: int
    include_shap: bool
    explanation_top_k: int
    prediction: Optional[PredictionResult]
    explanation: Optional[ExplanationResult]
    graph_analysis: Optional[GraphAnalysisResult]
    retrieved_docs: list
    investigation_report: str
    errors: list[str]


# ── High-Level Agent ─────────────────────────────────────────────────────────

class InvestigationAgent:
    """
    High-level interface for running fraud investigations.

    Services are stored on the agent instance (not in the graph state)
    to avoid serialization issues with LangGraph.
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
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct the LangGraph investigation workflow."""
        workflow = StateGraph(InvestigationState)

        # Add nodes — each is a bound method that has access to self.* services
        workflow.add_node("get_prediction", self._step_predict)
        workflow.add_node("get_explanation", self._step_explain)
        workflow.add_node("analyze_graph", self._step_graph)
        workflow.add_node("retrieve_knowledge", self._step_rag)
        workflow.add_node("generate_report", self._step_report)

        # Linear edges: predict → explain → graph → RAG → report
        workflow.set_entry_point("get_prediction")
        workflow.add_edge("get_prediction", "get_explanation")
        workflow.add_edge("get_explanation", "analyze_graph")
        workflow.add_edge("analyze_graph", "retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    # ── Workflow Steps ───────────────────────────────────────────────────────

    def _step_predict(self, state: InvestigationState) -> dict:
        """Step 1: Get GNN fraud prediction for the target node."""
        node_id = state["node_id"]
        try:
            result = self.prediction_service.predict_node(node_id)
            logger.info(
                "Step 1/5 [predict]: node=%d prob=%.4f label=%s",
                node_id, result.fraud_probability, result.label,
            )
            return {"prediction": result}
        except Exception as e:
            logger.error("Prediction failed: %s", e)
            return {"prediction": None, "errors": state.get("errors", []) + [f"Prediction: {e}"]}

    def _step_explain(self, state: InvestigationState) -> dict:
        """Step 2: Run GNNExplainer (or SHAP) on the target node."""
        node_id = state["node_id"]
        include_shap = state.get("include_shap", False)
        top_k = state.get("explanation_top_k", 10)

        try:
            if include_shap:
                result = self.explainability_service.explain_node_shap(node_id, top_k=top_k)
            else:
                result = self.explainability_service.explain_node_gnn(node_id, top_k=top_k)
            logger.info(
                "Step 2/5 [explain]: method=%s features=%d",
                result.method, len(result.top_features),
            )
            return {"explanation": result}
        except Exception as e:
            logger.error("Explanation failed: %s", e)
            return {"explanation": None, "errors": state.get("errors", []) + [f"Explanation: {e}"]}

    def _step_graph(self, state: InvestigationState) -> dict:
        """Step 3: Analyze the graph neighborhood of the target node."""
        node_id = state["node_id"]
        try:
            result = self.graph_analysis_service.analyze_node(node_id)
            logger.info(
                "Step 3/5 [graph]: degree=%d illicit_neighbors=%d flags=%s",
                result.total_degree, result.neighbors_1hop.illicit,
                result.structural_flags,
            )
            return {"graph_analysis": result}
        except Exception as e:
            logger.error("Graph analysis failed: %s", e)
            return {"graph_analysis": None, "errors": state.get("errors", []) + [f"Graph analysis: {e}"]}

    def _step_rag(self, state: InvestigationState) -> dict:
        """Step 4: Retrieve relevant fraud knowledge using RAG."""
        if self.retriever is None:
            logger.warning("Step 4/5 [RAG]: No retriever available, skipping")
            return {"retrieved_docs": []}

        prediction = state.get("prediction")
        explanation = state.get("explanation")
        graph_analysis = state.get("graph_analysis")

        top_features = []
        if explanation and hasattr(explanation, "top_features") and explanation.top_features:
            top_features = [f.feature_name for f in explanation.top_features[:5]]

        structural_flags = []
        if graph_analysis and hasattr(graph_analysis, "structural_flags"):
            structural_flags = graph_analysis.structural_flags

        neighbor_summary = ""
        if graph_analysis and hasattr(graph_analysis, "neighbors_1hop"):
            n = graph_analysis.neighbors_1hop
            neighbor_summary = (
                f"{n.total} neighbors, {n.illicit} illicit, {n.licit} licit, "
                f"{n.unknown} unknown"
            )

        query = self.retriever.build_query_from_context(
            prediction_label=prediction.label if prediction else "unknown",
            top_features=top_features,
            structural_flags=structural_flags,
            neighbor_summary=neighbor_summary,
        )

        try:
            docs = self.retriever.retrieve(query)
            logger.info("Step 4/5 [RAG]: Retrieved %d chunks", len(docs))
            return {"retrieved_docs": docs}
        except Exception as e:
            logger.error("RAG retrieval failed: %s", e)
            return {"retrieved_docs": [], "errors": state.get("errors", []) + [f"RAG: {e}"]}

    def _step_report(self, state: InvestigationState) -> dict:
        """Step 5: Generate the investigation report using the LLM."""
        node_id = state["node_id"]
        context = InvestigationContext(
            node_id=node_id,
            prediction=state.get("prediction"),
            explanation=state.get("explanation"),
            graph_analysis=state.get("graph_analysis"),
            errors=state.get("errors", []),
        )

        retrieved_docs = state.get("retrieved_docs", [])

        if self.report_generator is None:
            logger.warning("Step 5/5 [LLM]: No report generator, using fallback")
            sections = self.investigation_service.format_context_for_llm(context)
            report = ReportGenerator._fallback_report(context, sections)
        else:
            try:
                report = self.report_generator.generate_report(
                    context, retrieved_docs, self.investigation_service
                )
                logger.info("Step 5/5 [LLM]: Report generated (%d chars)", len(report))
            except Exception as e:
                logger.error("LLM report generation failed: %s", e)
                sections = self.investigation_service.format_context_for_llm(context)
                report = ReportGenerator._fallback_report(context, sections)
                return {
                    "investigation_report": report,
                    "errors": state.get("errors", []) + [f"LLM: {e}"],
                }

        return {"investigation_report": report}

    # ── Public API ───────────────────────────────────────────────────────────

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

        initial_state: InvestigationState = {
            "node_id": node_id,
            "include_shap": include_shap,
            "explanation_top_k": explanation_top_k,
            "errors": [],
        }

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

    def investigate_stream(self, node_id: int, include_shap: bool = False, explanation_top_k: int = 10):
        """Yields SSE-compatible progress and report chunks."""
        import json
        
        yield f'data: {json.dumps({"type": "status", "message": "Predicting node fraud probability..."})}\n\n'
        
        state: InvestigationState = {
            "node_id": node_id,
            "include_shap": include_shap,
            "explanation_top_k": explanation_top_k,
            "errors": [],
        }
        
        # Step 1: Predict
        state.update(self._step_predict(state))
        
        # Step 2: Explain
        yield f'data: {json.dumps({"type": "status", "message": "Extracting subgraph and explaining decisions..."})}\n\n'
        state.update(self._step_explain(state))
        
        # Step 3: Graph Analysis
        yield f'data: {json.dumps({"type": "status", "message": "Analyzing structural graph neighborhood..."})}\n\n'
        state.update(self._step_graph(state))
        
        # Step 4: RAG Retrieval
        yield f'data: {json.dumps({"type": "status", "message": "Retrieving fraud typologies from vector DB..."})}\n\n'
        state.update(self._step_rag(state))
        
        # Step 5: LLM Stream
        yield f'data: {json.dumps({"type": "status", "message": "Generating final investigation report..."})}\n\n'
        
        context = InvestigationContext(
            node_id=node_id,
            prediction=state.get("prediction"),
            explanation=state.get("explanation"),
            graph_analysis=state.get("graph_analysis"),
            errors=state.get("errors", []),
        )
        retrieved_docs = state.get("retrieved_docs", [])
        
        if self.report_generator is None:
            from src.llm.report_generator import ReportGenerator
            sections = self.investigation_service.format_context_for_llm(context)
            report = ReportGenerator._fallback_report(context, sections, retrieved_docs)
            yield f'data: {json.dumps({"type": "chunk", "text": report})}\n\n'
        else:
            for chunk in self.report_generator.generate_report_stream(
                context, retrieved_docs, self.investigation_service
            ):
                yield f'data: {json.dumps({"type": "chunk", "text": chunk})}\n\n'
        
        yield f'data: {json.dumps({"type": "done"})}\n\n'
