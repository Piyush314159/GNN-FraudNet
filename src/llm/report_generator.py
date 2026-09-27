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

    def _call_llm(self, prompt: str) -> str:
        """Call the Gemini API and return the generated text."""
        client = self._get_client()
        from google.genai import types

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )

        if response.text:
            return response.text.strip()
        else:
            raise RuntimeError("Empty response from LLM")

    @staticmethod
    def _fallback_report(
        context: InvestigationContext, sections: dict[str, str]
    ) -> str:
        """Generate a basic report without LLM when the API is unavailable."""
        lines = [
            "═══ INVESTIGATION REPORT (Auto-Generated — LLM Unavailable) ═══",
            "",
        ]

        if context.prediction:
            p = context.prediction
            risk = "HIGH" if p.fraud_probability > 0.7 else "MEDIUM" if p.fraud_probability > 0.3 else "LOW"
            lines.extend([
                f"RISK LEVEL: {risk}",
                f"Fraud Probability: {p.fraud_probability:.1%}",
                f"Predicted Label: {p.label}",
                f"Model Confidence: {p.confidence}",
                "",
            ])

        lines.extend([
            "KEY INDICATORS:",
            sections.get("explanation_section", "Unavailable"),
            "",
            "GRAPH CONTEXT:",
            sections.get("graph_section", "Unavailable"),
            "",
            "NOTE: LLM-powered analysis was unavailable. This report contains "
            "raw model outputs only. Set GEMINI_API_KEY in .env for full reports.",
        ])

        return "\n".join(lines)
