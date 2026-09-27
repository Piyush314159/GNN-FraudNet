"""
knowledge_base.py
=================
Builds and manages a ChromaDB vector store from the fraud knowledge documents.

On first run, reads markdown files from src/rag/documents/, chunks them,
embeds using sentence-transformers, and persists to disk.
On subsequent runs, loads from the persisted database.
"""
import logging
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE, RAG_DOCUMENTS_DIR

logger = logging.getLogger(__name__)

COLLECTION_NAME = "fraud_knowledge"


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count, respecting paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk; if overlap, keep the tail of the previous chunk
            if overlap > 0 and current_chunk:
                tail = current_chunk[-overlap:]
                current_chunk = f"{tail}\n\n{para}"
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _load_documents(docs_dir: Path) -> list[dict]:
    """Load all .md files from the documents directory."""
    documents = []
    if not docs_dir.exists():
        logger.warning("Documents directory not found: %s", docs_dir)
        return documents

    for md_file in sorted(docs_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        source = md_file.stem.replace("_", " ").title()
        chunks = _chunk_text(text, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{md_file.stem}_chunk_{i}",
                "text": chunk,
                "metadata": {
                    "source": source,
                    "file": md_file.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            })

    logger.info("Loaded %d chunks from %d documents", len(documents), len(list(docs_dir.glob("*.md"))))
    return documents


class KnowledgeBase:
    """Manages the ChromaDB vector store for fraud knowledge retrieval."""

    def __init__(
        self,
        persist_dir: str = CHROMA_DB_PATH,
        docs_dir: Path = RAG_DOCUMENTS_DIR,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.persist_dir = persist_dir
        self.docs_dir = docs_dir
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.client: chromadb.ClientAPI | None = None
        self.collection: chromadb.Collection | None = None

    def initialize(self) -> None:
        """Load or build the knowledge base."""
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Check if collection already exists with documents
        existing_collections = [c.name for c in self.client.list_collections()]
        if COLLECTION_NAME in existing_collections:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embedding_fn,
            )
            count = self.collection.count()
            if count > 0:
                logger.info(
                    "Loaded existing knowledge base: %d chunks", count
                )
                return

        # Build from scratch
        logger.info("Building knowledge base from %s", self.docs_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"description": "GNN-FraudNet fraud knowledge base"},
        )
        self._ingest_documents()

    def _ingest_documents(self) -> None:
        """Load documents, chunk, embed, and store in ChromaDB."""
        documents = _load_documents(self.docs_dir)
        if not documents:
            logger.warning("No documents to ingest")
            return

        ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]
        metadatas = [d["metadata"] for d in documents]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Ingested %d chunks into knowledge base", len(documents))

    def rebuild(self) -> None:
        """Force rebuild the knowledge base from documents."""
        if self.client is None:
            self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Delete existing collection if present
        existing_collections = [c.name for c in self.client.list_collections()]
        if COLLECTION_NAME in existing_collections:
            self.client.delete_collection(COLLECTION_NAME)

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"description": "GNN-FraudNet fraud knowledge base"},
        )
        self._ingest_documents()

    def get_collection(self) -> chromadb.Collection:
        """Return the initialized collection."""
        if self.collection is None:
            raise RuntimeError("KnowledgeBase not initialized. Call .initialize() first.")
        return self.collection
