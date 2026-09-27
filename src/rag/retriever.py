"""
retriever.py
============
Semantic retrieval over the fraud knowledge base.

Given a query (typically constructed from an investigation context),
returns the most relevant document chunks with relevance scores.
"""
import logging
from dataclasses import dataclass

from src.config import RAG_TOP_K
from src.rag.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A single retrieved document chunk with metadata."""

    text: str
    source: str
    relevance_score: float
    chunk_id: str


class Retriever:
    """Performs semantic search over the fraud knowledge base."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[RetrievedDocument]:
        """
        Search the knowledge base for documents relevant to the query.

        Args:
            query: Natural language query constructed from investigation context.
            top_k: Number of chunks to retrieve.

        Returns:
            List of RetrievedDocument sorted by relevance (best first).
        """
        collection = self.kb.get_collection()

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                doc_id = results["ids"][0][i] if results["ids"] else f"unknown_{i}"

                # ChromaDB returns L2 distance; convert to a 0-1 relevance score
                # Lower distance = more relevant
                relevance = 1.0 / (1.0 + distance)

                documents.append(
                    RetrievedDocument(
                        text=doc_text,
                        source=metadata.get("source", "Unknown"),
                        relevance_score=round(relevance, 4),
                        chunk_id=doc_id,
                    )
                )

        logger.info("Retrieved %d chunks for query: %s...", len(documents), query[:80])
        return documents

    def build_query_from_context(
        self,
        prediction_label: str,
        top_features: list[str],
        structural_flags: list[str],
        neighbor_summary: str,
    ) -> str:
        """
        Construct a search query from investigation context.

        This converts structured investigation data into a natural language
        query optimized for embedding-based retrieval.
        """
        parts = []

        if prediction_label in ("illicit", "uncertain"):
            parts.append(f"Bitcoin transaction flagged as {prediction_label}.")

        if top_features:
            features_str = ", ".join(top_features[:5])
            parts.append(f"Important features: {features_str}.")

        if structural_flags:
            flags_str = ", ".join(structural_flags)
            parts.append(f"Graph patterns detected: {flags_str}.")

        if neighbor_summary:
            parts.append(f"Neighborhood: {neighbor_summary}.")

        # Add a general fraud detection context
        parts.append("Fraud detection graph neural network explanation.")

        return " ".join(parts)
