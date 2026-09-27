"""
config.py
=========
Centralized configuration for GNN-FraudNet.

All settings are loaded from environment variables with sensible defaults.
Use a .env file in the project root for local development.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (no-op if file doesn't exist)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = _PROJECT_ROOT
MODEL_PATH = os.getenv("MODEL_PATH", "results/best_graphsage.pt")
DATA_PATH = os.getenv("DATA_PATH", "data/processed/elliptic_graph.pt")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "knowledge_base/chroma_db")
RAG_DOCUMENTS_DIR = PROJECT_ROOT / "src" / "rag" / "documents"

# ── Model ────────────────────────────────────────────────────────────────────

NUM_FEATURES = 166
HIDDEN_CHANNELS = 64
NUM_CLASSES = 2

# ── LLM ──────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── RAG ──────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_CHUNK_SIZE = 500          # characters per chunk
RAG_CHUNK_OVERLAP = 50        # overlap between chunks

# ── App ──────────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# ── Thresholds ───────────────────────────────────────────────────────────────

FRAUD_THRESHOLD_HIGH = 0.7     # above → illicit
FRAUD_THRESHOLD_LOW = 0.3      # below → licit, between → uncertain
