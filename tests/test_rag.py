"""
test_rag.py
===========
Tests for the RAG knowledge base and retriever.
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest


class TestKnowledgeBase:

    def test_chunk_text(self):
        from src.rag.knowledge_base import _chunk_text

        text = "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4."
        chunks = _chunk_text(text, chunk_size=20, overlap=0)
        assert len(chunks) >= 2
        assert all(len(c) > 0 for c in chunks)

    def test_chunk_text_empty(self):
        from src.rag.knowledge_base import _chunk_text

        chunks = _chunk_text("", chunk_size=100, overlap=0)
        assert len(chunks) == 0

    def test_load_documents(self):
        from src.rag.knowledge_base import _load_documents
        from src.config import RAG_DOCUMENTS_DIR

        docs = _load_documents(RAG_DOCUMENTS_DIR)
        assert len(docs) > 0
        assert all("text" in d for d in docs)
        assert all("metadata" in d for d in docs)
        assert all("id" in d for d in docs)

    def test_documents_have_sources(self):
        from src.rag.knowledge_base import _load_documents
        from src.config import RAG_DOCUMENTS_DIR

        docs = _load_documents(RAG_DOCUMENTS_DIR)
        sources = set(d["metadata"]["source"] for d in docs)
        expected_sources = {
            "Fraud Typologies",
            "Graph Patterns",
            "Feature Definitions",
            "Model Documentation",
        }
        assert sources == expected_sources


class TestRetriever:

    def test_build_query_from_context(self):
        from src.rag.retriever import Retriever

        # We only need the query builder, which doesn't need a real KB
        class MockKB:
            def get_collection(self):
                return None

        retriever = Retriever(MockKB())
        query = retriever.build_query_from_context(
            prediction_label="illicit",
            top_features=["feature_14", "feature_97"],
            structural_flags=["fan_out_pattern", "high_illicit_concentration"],
            neighbor_summary="12 neighbors, 7 illicit",
        )
        assert "illicit" in query
        assert "feature_14" in query
        assert "fan_out_pattern" in query

    def test_build_query_empty_context(self):
        from src.rag.retriever import Retriever

        class MockKB:
            def get_collection(self):
                return None

        retriever = Retriever(MockKB())
        query = retriever.build_query_from_context(
            prediction_label="licit",
            top_features=[],
            structural_flags=[],
            neighbor_summary="",
        )
        # Should still produce a valid query string
        assert len(query) > 0
