"""
report_generator.py
===================
LLM-based investigation report generation using Google Gemini.

Takes structured investigation context + RAG results, renders the prompt,
calls the Gemini API, and returns the generated report.
"""
import logging
from typing import Optional

from src.config import GEMINI_API_KEY, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE
from src.llm.prompts import INVESTIGATION_PROMPT
from src.rag.retriever import RetrievedDocument
from src.services.investigation import InvestigationContext, InvestigationService

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates investigation reports using an LLM."""

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = LLM_MODEL):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy-init the Gemini client."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. "
                    "Get a free key at https://aistudio.google.com and add it to .env"
                )
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_report(
        self,
        context: InvestigationContext,
        retrieved_docs: list[RetrievedDocument],
        investigation_service: InvestigationService,
    ) -> str:
        """
        Generate a natural-language investigation report.

        Args:
            context: Complete investigation context from the pipeline.
            retrieved_docs: RAG-retrieved fraud knowledge chunks.
            investigation_service: Service for formatting context into prompt sections.

        Returns:
            Generated investigation report as a string.
        """
        # Format context into prompt-ready sections
        sections = investigation_service.format_context_for_llm(context)

        # Format RAG results
        rag_lines = []
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs, 1):
                rag_lines.append(
                    f"[Source: {doc.source} | Relevance: {doc.relevance_score:.2f}]\n"
                    f"{doc.text}"
                )
            rag_section = "\n\n---\n\n".join(rag_lines)
        else:
            rag_section = "No relevant fraud intelligence retrieved."

        # Determine explanation method
        explanation_method = "Unknown"
        if context.explanation:
            explanation_method = context.explanation.method.upper().replace("_", " ")

        # Render prompt
        prompt = INVESTIGATION_PROMPT.format(
            prediction_section=sections.get("prediction_section", "Unavailable"),
            explanation_section=sections.get("explanation_section", "Unavailable"),
            graph_section=sections.get("graph_section", "Unavailable"),
            rag_section=rag_section,
            explanation_method=explanation_method,
        )

        # Call LLM
        try:
            report = self._call_llm(prompt)
            return report
        except Exception as e:
            logger.error("LLM report generation failed: %s", e)
            return self._fallback_report(context, sections)

    def generate_report_stream(
        self,
        context: InvestigationContext,
        retrieved_docs: list[RetrievedDocument],
        investigation_service: InvestigationService,
    ):
        """Yields chunks of the generated report."""
        sections = investigation_service.format_context_for_llm(context)
        
        rag_lines = []
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs, 1):
                rag_lines.append(f"[Source: {doc.source} | Relevance: {doc.relevance_score:.2f}]\n{doc.text}")
            rag_section = "\n\n---\n\n".join(rag_lines)
        else:
            rag_section = "No relevant fraud intelligence retrieved."

        explanation_method = context.explanation.method.upper().replace("_", " ") if context.explanation else "Unknown"

        prompt = INVESTIGATION_PROMPT.format(
            prediction_section=sections.get("prediction_section", "Unavailable"),
            explanation_section=sections.get("explanation_section", "Unavailable"),
            graph_section=sections.get("graph_section", "Unavailable"),
            rag_section=rag_section,
            explanation_method=explanation_method,
        )

        try:
            for chunk in self._call_llm_stream(prompt):
                yield chunk
        except Exception as e:
            logger.error("LLM report stream generation failed: %s", e)
            yield self._fallback_report(context, sections)

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the Gemini API with retry logic for transient errors."""
        import time

        client = self._get_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_budget=1024,
            ),
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                if response.text:
                    return response.text.strip()
                else:
                    raise RuntimeError("Empty response from LLM")
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Retry on 503 (overloaded) or 429 (rate limit)
                if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "LLM attempt %d/%d failed (retrying in %ds): %s",
                        attempt + 1, max_retries, wait, e,
                    )
                    time.sleep(wait)
                else:
                    raise  # Non-retryable error

        raise last_error

    def _call_llm_stream(self, prompt: str, max_retries: int = 3):
        import time
        client = self._get_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                response_stream = client.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str:
                    wait = 2 ** attempt
                    logger.warning("LLM stream attempt %d/%d failed: %s", attempt + 1, max_retries, e)
                    time.sleep(wait)
                else:
                    raise

        raise last_error

    @staticmethod
    def _fallback_report(
        context: InvestigationContext, sections: dict[str, str], retrieved_docs: list = None
    ) -> str:
        """Generate a highly structured report without LLM when the API is unavailable."""
        lines = []

        # 1. RISK ASSESSMENT
        lines.append("### 1. RISK ASSESSMENT\n")
        if context.prediction:
            p = context.prediction
            risk = "HIGH" if p.fraud_probability > 0.7 else "MEDIUM" if p.fraud_probability > 0.3 else "LOW"
            lines.append(f"- **Severity Level:** {risk}")
            lines.append(
                f"- **Model Evidence:** Node ID {context.node_id} has an illicit fraud probability of **{p.fraud_probability:.2%}** (licit: {1-p.fraud_probability:.2%}). The predicted label is explicitly designated as **\"{p.label}\"** with **{p.confidence.lower()}** model confidence."
            )
        else:
            lines.append("- Prediction unavailable.")
        lines.append("")

        # 2. KEY INDICATORS
        lines.append("### 2. KEY INDICATORS\n")
        if context.explanation and not context.explanation.error:
            e = context.explanation
            local_feats = [f for f in e.top_features if int(f.feature_name.split("_")[1]) <= 93]
            agg_feats = [f for f in e.top_features if int(f.feature_name.split("_")[1]) > 93]

            lines.append(f"- **Top Contributing Features** (all direction: `increases_fraud`):")
            
            if local_feats:
                feat_str = ", ".join([f"{f.feature_name} ({f.importance:.4f})" for f in local_feats])
                lines.append(f"  - *Local Features ({len(local_feats)} of top {len(e.top_features)}):* {feat_str}.")
            if agg_feats:
                feat_str = ", ".join([f"{f.feature_name} ({f.importance:.4f})" for f in agg_feats])
                lines.append(f"  - *Aggregated Neighborhood Features ({len(agg_feats)} of top {len(e.top_features)}):* {feat_str}.")
        else:
            lines.append("- Explanation features unavailable.")

        if context.graph_analysis:
            g = context.graph_analysis
            lines.append(f"- **Graph Structure:** In-degree is {g.in_degree}, out-degree is {g.out_degree} (total degree: {g.total_degree}).")
            flags = ", ".join(g.structural_flags) if g.structural_flags else "none"
            lines.append(f"- **Structural Flags:** `{flags}` is present.")
        lines.append("")

        # 3. PATTERN ANALYSIS
        lines.append("### 3. PATTERN ANALYSIS\n")
        
        if context.explanation and not context.explanation.error:
            e = context.explanation
            local_count = sum(1 for f in e.top_features if int(f.feature_name.split("_")[1]) <= 93)
            agg_count = len(e.top_features) - local_count
            
            if local_count >= agg_count:
                lines.append(f"- **Model Evidence:** GNN Explainer identifies that the prediction is primarily driven by local transaction properties ({local_count / len(e.top_features):.0%} of top features).")
            else:
                lines.append(f"- **Model Evidence:** GNN Explainer identifies that the prediction is primarily driven by aggregated neighborhood context ({agg_count / len(e.top_features):.0%} of top features).")
        else:
            lines.append("- **Model Evidence:** The prediction is primarily driven by the extracted features and neighborhood context.")
            
        if context.graph_analysis:
            g = context.graph_analysis
            flags = ", ".join(g.structural_flags) if g.structural_flags else "none"
            lines.append(f"- **Interpretation:** Per intelligence documentation, high-importance features indicate the transaction possesses unusual properties that trigger fraud signals independently of dense graph context. Although the structural flag designates a `{flags}`, the immediate neighborhood shows an out-degree of only {g.out_degree}, and there are {g.neighbors_1hop.total + g.neighbors_2hop.total} total labeled neighbors. No specific complex money-laundering typologies (such as peeling chains or multi-layer mixing) can be fully confirmed without the full AI capabilities.")
        else:
            lines.append("- **Interpretation:** *LLM-powered analysis was temporarily unavailable due to API rate limits. This report contains the raw foundational model outputs and heuristics without the advanced textual interpretation.*")
        lines.append("")

        # 4. LIMITATIONS & UNCERTAINTY
        lines.append("### 4. LIMITATIONS & UNCERTAINTY\n")
        
        if context.prediction:
            p = context.prediction
            lines.append(f"- **Prediction Uncertainty:** The model's classification is explicitly categorized as **{p.label}** with **{p.confidence.lower()}** confidence.")
            
        if context.graph_analysis:
            g = context.graph_analysis
            lines.append(f"- **Neighborhood Data Void:** There are {g.neighbors_1hop.total} total 1-hop and {g.neighbors_2hop.total} total 2-hop neighbors recorded in the subgraph ({g.neighbors_1hop.licit + g.neighbors_2hop.licit} licit, {g.neighbors_1hop.illicit + g.neighbors_2hop.illicit} illicit, {g.neighbors_1hop.unknown + g.neighbors_2hop.unknown} unknown labeled neighbors). Ground truth for Node {context.node_id} itself is {g.ground_truth_label}.")
            
        lines.append("- **Anonymization & Scope:** All 166 features are anonymized, preventing direct verification of actual values (e.g., fee rates, exact BTC volumes). Additionally, the model is limited to a static 2-hop receptive field without temporal sequence tracking.")
        
        # 5. RECOMMENDED ACTIONS
        lines.append("")
        lines.append("### 5. RECOMMENDED ACTIONS\n")
        lines.append("- **Action 1 - Feature Validation:** Cross-reference the highly-weighted local features (e.g. amounts, timestamps, or velocity metrics) with the internal exchange database to check for known illicit thresholds.")
        
        if context.graph_analysis:
            g = context.graph_analysis
            if g.out_degree > 0:
                lines.append(f"- **Action 2 - Counterparty KYC:** Investigate the {g.out_degree} receiving counterparties (out-degree). Determine if they belong to unregulated exchanges or known darknet markets.")
            elif g.in_degree > 0:
                lines.append(f"- **Action 3 - Source of Funds:** Trace the {g.in_degree} incoming transactions to verify the source of funds. Check if any incoming addresses are blacklisted.")
            else:
                lines.append("- **Action 2 - Neighborhood Expansion:** Expand the subgraph search beyond 2-hops to look for connections to known sanctioned entities.")
        
        lines.append("- **Action 4 - Watchlist:** Temporarily flag this transaction and related wallets for enhanced monitoring pending manual review.")

        # 6. INVESTIGATOR Q&A
        lines.append("")
        lines.append("### 6. INVESTIGATOR Q&A\n")
        
        if retrieved_docs and len(retrieved_docs) > 0:
            lines.append("- **Follow-up Inquiry 1:** Based on retrieved intelligence, does this node exhibit behavior similar to the known typology: *\"" + retrieved_docs[0].text[:80].replace('\n', ' ') + "...\"*?")
            if len(retrieved_docs) > 1:
                lines.append(f"- **Follow-up Inquiry 2:** Could the structural flags (`{flags if 'flags' in locals() else 'none'}`) be indicative of the pattern described in RAG Document [{retrieved_docs[1].chunk_id}]?")
        else:
            lines.append("- **Follow-up Inquiry:** What real-world entities do the anomalous high-importance features correspond to?")
            lines.append("- **Follow-up Inquiry:** Are there any known darknet market addresses within a 3-hop radius of this node?")

        return "\n".join(lines)
