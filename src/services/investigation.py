"""
investigation.py
================
Investigation context builder — orchestrates prediction, explainability,
and graph analysis into a single structured context for LLM consumption.

This is a pure aggregation layer with no ML or LLM logic.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.services.explainability import ExplanationResult
from src.services.graph_analysis import GraphAnalysisResult
from src.services.prediction import PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class InvestigationContext:
    """Complete context for a fraud investigation, prior to LLM generation."""

    node_id: int
    prediction: Optional[PredictionResult] = None
    explanation: Optional[ExplanationResult] = None
    graph_analysis: Optional[GraphAnalysisResult] = None
    retrieved_context: list[dict] = field(default_factory=list)
    investigation_report: str = ""
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class InvestigationService:
    """Builds investigation context by orchestrating all services."""

    def __init__(self, prediction_service, explainability_service, graph_analysis_service):
        self.prediction_svc = prediction_service
        self.explain_svc = explainability_service
        self.graph_svc = graph_analysis_service

    def build_context(
        self,
        node_id: int,
        include_shap: bool = False,
        explanation_top_k: int = 10,
    ) -> InvestigationContext:
        """
        Build a complete investigation context for a given node.

        Args:
            node_id: The graph node to investigate.
            include_shap: If True, run SHAP in addition to GNNExplainer (slower).
            explanation_top_k: Number of top features to include.

        Returns:
            InvestigationContext with prediction, explanation, and graph data.
        """
        start_time = time.time()
        context = InvestigationContext(node_id=node_id)

        # Step 1: Get GNN prediction
        try:
            context.prediction = self.prediction_svc.predict_node(node_id)
            logger.info(
                "Node %d: fraud_prob=%.4f label=%s",
                node_id,
                context.prediction.fraud_probability,
                context.prediction.label,
            )
        except Exception as e:
            logger.error("Prediction failed for node %d: %s", node_id, e)
            context.errors.append(f"Prediction failed: {e}")

        # Step 2: Get model explanation
        try:
            if include_shap:
                context.explanation = self.explain_svc.explain_node_shap(
                    node_id, top_k=explanation_top_k
                )
            else:
                context.explanation = self.explain_svc.explain_node_gnn(
                    node_id, top_k=explanation_top_k
                )
        except Exception as e:
            logger.error("Explanation failed for node %d: %s", node_id, e)
            context.errors.append(f"Explanation failed: {e}")

        # Step 3: Analyze graph neighborhood
        try:
            context.graph_analysis = self.graph_svc.analyze_node(node_id)
        except Exception as e:
            logger.error("Graph analysis failed for node %d: %s", node_id, e)
            context.errors.append(f"Graph analysis failed: {e}")

        context.duration_seconds = time.time() - start_time
        return context

    def format_context_for_llm(self, context: InvestigationContext) -> dict[str, str]:
        """
        Format investigation context into prompt-ready text sections.

        Returns a dict with keys: prediction_section, explanation_section,
        graph_section — ready to be inserted into the LLM prompt template.
        """
        sections = {}

        # Prediction section
        if context.prediction:
            p = context.prediction
            sections["prediction_section"] = (
                f"- Node ID: {p.node_id}\n"
                f"- Fraud Probability: {p.fraud_probability:.4f} ({p.fraud_probability:.1%})\n"
                f"- Predicted Label: {p.label}\n"
                f"- Model Confidence: {p.confidence}\n"
                f"- Raw Probabilities: licit={p.probabilities[0]:.4f}, illicit={p.probabilities[1]:.4f}"
            )
        else:
            sections["prediction_section"] = "Prediction unavailable."

        # Explanation section
        if context.explanation and not context.explanation.error:
            e = context.explanation
            lines = [f"Method: {e.method}", ""]
            lines.append("Top contributing features:")
            for feat in e.top_features:
                lines.append(
                    f"  {feat.rank}. {feat.feature_name}: "
                    f"importance={feat.importance:.4f}, "
                    f"direction={feat.direction}"
                )

            # Distinguish local vs. aggregated features
            local_count = sum(
                1 for f in e.top_features if int(f.feature_name.split("_")[1]) <= 93
            )
            agg_count = len(e.top_features) - local_count
            lines.append("")
            lines.append(
                f"Of the top {len(e.top_features)} features: "
                f"{local_count} are local transaction features (1-93), "
                f"{agg_count} are aggregated neighborhood features (94-166)."
            )
            if agg_count > local_count:
                lines.append(
                    "The prediction is primarily driven by neighborhood context "
                    "(the graph structure matters more than the transaction itself)."
                )
            else:
                lines.append(
                    "The prediction is primarily driven by the transaction's own features "
                    "(the transaction itself is unusual)."
                )
            sections["explanation_section"] = "\n".join(lines)
        elif context.explanation and context.explanation.error:
            sections["explanation_section"] = (
                f"Explanation failed: {context.explanation.error}"
            )
        else:
            sections["explanation_section"] = "Explanation unavailable."

        # Graph section
        if context.graph_analysis:
            g = context.graph_analysis
            n1 = g.neighbors_1hop
            lines = [
                f"- Ground truth label in dataset: {g.ground_truth_label}",
                f"- In-degree: {g.in_degree}, Out-degree: {g.out_degree}, Total degree: {g.total_degree}",
                "",
                "1-hop Neighbors:",
                f"  Total: {n1.total}, Illicit: {n1.illicit}, Licit: {n1.licit}, Unknown: {n1.unknown}",
                f"  Illicit ratio (among labeled): {n1.illicit_ratio:.2%}",
            ]
            if g.neighbors_2hop:
                n2 = g.neighbors_2hop
                lines.extend([
                    "",
                    "2-hop Neighbors:",
                    f"  Total: {n2.total}, Illicit: {n2.illicit}, Licit: {n2.licit}, Unknown: {n2.unknown}",
                    f"  Illicit ratio (among labeled): {n2.illicit_ratio:.2%}",
                ])
            if g.structural_flags:
                lines.extend(["", f"Structural flags: {', '.join(g.structural_flags)}"])
            else:
                lines.extend(["", "No notable structural flags detected."])

            sections["graph_section"] = "\n".join(lines)
        else:
            sections["graph_section"] = "Graph analysis unavailable."

        return sections
