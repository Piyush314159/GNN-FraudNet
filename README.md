# GNN-FraudNet

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch_Geometric-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langgraph&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/AWS_EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white"/>
</p>

<p align="center">
  <b>AI-Powered Fraud Detection & Investigation Platform</b><br/>
  GNN (GraphSAGE/GAT) · Explainable AI (SHAP/GNNExplainer) · RAG · LLM Reports · LangGraph Agent · FastAPI · Docker · AWS
</p>

---

## 🧠 What Is This?

An end-to-end fraud detection and investigation system that combines **Graph Neural Networks** with **LLM-powered investigation reports**.

```
Traditional ML:   "Is THIS transaction suspicious?"   → looks at 166 features of ONE transaction
GNN (This model): "Is this suspicious GIVEN context?" → looks at the transaction AND its neighbors
This system:      "WHY is it suspicious?"              → explains, retrieves knowledge, generates report
```

The system goes beyond prediction — it provides a full **investigation pipeline** that an analyst can use to understand and act on fraud detections.

---

## 🏗️ Architecture

```
                          ┌─────────────────────┐
                          │     Transaction      │
                          │      (Node ID)       │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   GraphSAGE / GAT   │
                          │   Fraud Prediction   │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Investigation      │
                          │   Context Builder    │
                          └────┬────────────┬───┘
                               │            │
                    ┌──────────▼──┐   ┌─────▼──────────┐
                    │SHAP / GNN   │   │RAG Retrieval    │
                    │Explainer    │   │Fraud Knowledge  │
                    └──────────┬──┘   └─────┬──────────┘
                               │            │
                          ┌────▼────────────▼───┐
                          │      LLM API        │
                          │  (Google Gemini)     │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Natural-Language    │
                          │  Investigation       │
                          │  Report              │
                          └─────────────────────┘
```

### Agentic Workflow (LangGraph)

```
predict → explain → analyze_graph → retrieve_knowledge → generate_report
```

Each step is a LangGraph node. State accumulates as the workflow progresses.

---

## 📊 Dataset

**Elliptic Bitcoin Dataset** — one of the few real-world labeled cryptocurrency fraud datasets.

| Property | Value |
|----------|-------|
| Nodes (transactions) | 203,769 |
| Edges (BTC flows) | 234,355 |
| Features per node | 166 |
| Illicit (fraud) | 4,545 |
| Licit (legit) | 42,019 |
| Unknown | 157,205 |

> 📥 Download → [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
> Place the 3 CSVs in `data/raw/`

---

## 📈 Results

Test-set metrics from a 70/15/15 split of the labeled nodes:

| Model | F1 (Illicit) | PR-AUC | Accuracy | Uses Graph? |
|-------|:-----------:|:------:|:--------:|:-----------:|
| Logistic Regression | 0.602 | 0.757 | 0.880 | ✗ |
| Random Forest | 0.950 | 0.983 | 0.991 | ✗ |
| **XGBoost** | **0.954** | **0.987** | **0.991** | ✗ |
| **GraphSAGE** | **0.739** | **0.917** | **0.936** | **✓** |
| GAT | 0.449 | 0.615 | 0.779 | ✓ |

> **Honest takeaway:** Tree-based baselines (XGBoost, RF) beat GNNs on this dataset because the 166 engineered features already encode neighborhood information. GraphSAGE still demonstrates graph-aware learning and clearly beats linear models.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Gemini API key](https://aistudio.google.com) (free — for LLM reports)

### 1. Clone & Setup

```bash
git clone https://github.com/Piyush314159/GNN-FraudNet.git
cd GNN-FraudNet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install python-dotenv chromadb sentence-transformers google-genai langgraph
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000** → interactive dashboard.

### 4. Test the Investigation Endpoint

```bash
# Investigate a known illicit node
curl -X POST http://localhost:8000/investigate/42 \
  -H "Content-Type: application/json" \
  -d '{"include_shap": false, "explanation_top_k": 10}'
```

### Docker

```bash
# Build & run
docker-compose up --build

# Or manually:
docker build -t gnn-fraudnet .
docker run -p 8000:8000 --env-file .env gnn-fraudnet
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive fraud detection dashboard |
| `GET` | `/health` | Health check with service availability |
| `POST` | `/predict` | Predict from 166 raw features |
| `GET` | `/predict/{node_id}` | Predict for a known graph node |
| `POST` | `/investigate/{node_id}` | **Full investigation** — prediction + explanation + RAG + LLM report |
| `GET` | `/node/{node_id}` | Graph info — degree, neighbors, structural flags |

### Investigation Response

```json
{
  "node_id": 42,
  "prediction": {
    "fraud_probability": 0.87,
    "label": "illicit",
    "confidence": "high",
    "probabilities": [0.13, 0.87]
  },
  "explanation": {
    "method": "gnn_explainer",
    "top_features": [
      {"feature_name": "feature_14", "importance": 0.23, "direction": "increases_fraud", "rank": 1}
    ]
  },
  "graph_analysis": {
    "total_degree": 12,
    "neighbors_1hop": {"total": 12, "illicit": 7, "licit": 3, "unknown": 2},
    "structural_flags": ["high_illicit_concentration"]
  },
  "retrieved_context": [
    {"text": "Mixing services are designed to obscure...", "source": "Fraud Typologies", "relevance_score": 0.82}
  ],
  "investigation_report": "## RISK ASSESSMENT\n...",
  "metadata": {
    "duration_seconds": 3.45,
    "model_name": "GraphSAGE",
    "explanation_method": "gnn_explainer"
  }
}
```

---

## 📁 Project Structure

```
GNN-FraudNet/
│
├── src/
│   ├── config.py                    ← centralized configuration
│   ├── models/
│   │   ├── graphsage.py             ← GraphSAGE model
│   │   ├── gat.py                   ← GAT model
│   │   └── baseline.py              ← LogReg / RF / XGBoost
│   ├── services/
│   │   ├── prediction.py            ← model inference service
│   │   ├── explainability.py        ← SHAP / GNNExplainer wrapper
│   │   ├── graph_analysis.py        ← neighbor & structural analysis
│   │   └── investigation.py         ← investigation context builder
│   ├── rag/
│   │   ├── knowledge_base.py        ← ChromaDB vector store
│   │   ├── retriever.py             ← semantic search
│   │   └── documents/               ← fraud knowledge markdown files
│   ├── llm/
│   │   ├── prompts.py               ← anti-hallucination prompt templates
│   │   └── report_generator.py      ← Gemini API integration
│   ├── agents/
│   │   └── investigation_agent.py   ← LangGraph workflow
│   ├── data_loader.py               ← CSV → PyG Data
│   ├── graph_builder.py             ← normalization & masks
│   ├── train.py                     ← training loop
│   ├── evaluate.py                  ← metrics & comparison
│   └── explain.py                   ← GNNExplainer + SHAP
│
├── api/
│   ├── main.py                      ← FastAPI app + all endpoints
│   ├── schema.py                    ← Pydantic request/response models
│   └── static/index.html            ← interactive dashboard
│
├── tests/                           ← pytest test suite
├── notebooks/                       ← research pipeline (4 notebooks)
├── data/                            ← raw CSVs + processed graph
├── results/                         ← trained models + metrics
│
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml         ← GitHub Actions CI
├── .env.example                     ← environment template
└── README.md
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_prediction.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov=api
```

Tests use synthetic 50-node graphs — no need for the real dataset.

---

## ☁️ AWS Deployment

### Architecture

```
User → AWS EC2 (t3.medium) → Docker → FastAPI → GNN + RAG + Gemini API
```

### Steps

1. **Launch EC2** (Ubuntu 22.04, t3.medium, 30GB EBS)
2. **Install Docker**:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose
   sudo usermod -aG docker $USER
   ```
3. **Transfer files**:
   ```bash
   scp -r GNN-FraudNet/ ec2-user@<IP>:~/
   ```
4. **Configure**:
   ```bash
   cp .env.example .env
   # Add GEMINI_API_KEY
   ```
5. **Run**:
   ```bash
   docker-compose up -d
   ```
6. **Security Group**: Allow inbound TCP on port 8000 (or 80 with nginx reverse proxy)

> **Cost**: t3.medium ≈ $0.042/hr (~$30/month). Use a spot instance for ~$10/month.

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| GNN Framework | PyTorch Geometric |
| Deep Learning | PyTorch 2.4.0 |
| Baseline Models | XGBoost · Random Forest · scikit-learn |
| Explainability | SHAP · GNNExplainer |
| RAG | ChromaDB · sentence-transformers |
| LLM | Google Gemini (gemini-2.0-flash) |
| Agentic Workflow | LangGraph |
| API | FastAPI · Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (cyberpunk dashboard) |
| Deployment | Docker · Docker Compose · AWS EC2 |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## 🔬 Running the Full ML Pipeline

The ML pipeline was built as four notebooks. Run them **in order**:

```bash
jupyter lab notebooks/
```

| # | Notebook | What It Does |
|---|----------|--------------|
| 1 | `01_eda.ipynb` | Dataset exploration |
| 2 | `02_graph_construction.ipynb` | Build PyG Data object |
| 3 | `03_baseline_models.ipynb` | Train LogReg / RF / XGBoost |
| 4 | `04_gnn_training.ipynb` | Train GraphSAGE / GAT + explainability |

> **Tip:** All notebooks have pre-rendered outputs — open them to see results without re-executing.

---

## 🔗 Background

This project bridges **particle physics and fintech**.
The GNN message-passing framework applied here originates from work on the
[Belle II experiment](https://www.belle2.org/) — where GNNs reconstruct
particle decay trees from detector hits.
The same structural reasoning applies to financial transaction graphs.

---

<p align="center">
  Made by <a href="https://piyush314159.github.io">Piyush</a> · MSc Physics · IIT Hyderabad
</p>
