# GNN-FraudNet

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch_Geometric-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
</p>

<p align="center">
  <b>Graph Neural Network based Bitcoin fraud detection with interactive web dashboard.</b><br/>
  GraphSAGE · GAT · SHAP + GNNExplainer · REST API · Docker · Interactive UI
</p>

---

## 🧠 What Is This?

Traditional ML treats each transaction in isolation. **GNN-FraudNet exploits the graph** — a transaction surrounded by illicit neighbors is far more likely to be illicit itself.

```
 Traditional ML:   "Is THIS transaction suspicious?"   → looks at 166 features of ONE transaction
 GNN (This model): "Is this suspicious GIVEN context?" → looks at the transaction AND its neighbors
```

This project builds a complete pipeline: **data → training → evaluation → explainability → API → Docker → interactive dashboard**.

---

## 🏗️ Architecture

```
 ┌─────────────────────────────────────────────────────────────┐
 │                    GNN-FraudNet Pipeline                     │
 ├─────────────────────────────────────────────────────────────┤
 │                                                             │
 │  Elliptic Bitcoin Dataset (203K nodes · 234K edges)         │
 │                      ↓                                      │
 │  Node Features (166) + Edge Index                           │
 │                      ↓                                      │
 │  Normalize · Train/Val/Test Masks · Class Weights           │
 │                      ↓                                      │
 │  ┌────────────────────────────────────────────────┐         │
 │  │  GraphSAGE  ·  GAT  ·  XGBoost  ·  LR  ·  RF │         │
 │  │        5 models trained & compared             │         │
 │  └────────────────────────────────────────────────┘         │
 │                      ↓                                      │
 │  Fraud Probability per Transaction                          │
 │                      ↓                                      │
 │  SHAP + GNNExplainer  ← "Why was this flagged?"             │
 │                      ↓                                      │
 │  FastAPI + Interactive Dashboard + Docker                    │
 │                                                             │
 └─────────────────────────────────────────────────────────────┘
```

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

> **Honest takeaway:** Tree-based baselines (XGBoost, RF) beat GNNs on this dataset because the 166 engineered features already encode neighborhood information. GraphSAGE still demonstrates graph-aware learning and clearly beats linear models. See `notebooks/04_gnn_training.ipynb` for full analysis.

---

## 🚀 Quick Start

### Option 1: Local (Development)

```bash
# 1. Clone & setup
git clone https://github.com/Piyush314159/GNN-FraudNet.git
cd GNN-FraudNet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the API + interactive dashboard
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000** → interactive fraud detection dashboard.

### Option 2: Docker (Production)

```bash
# 1. Download torch wheel locally (avoids Docker network timeout)
pip download torch==2.4.0 \
  --platform manylinux2014_aarch64 \
  --python-version 311 \
  --only-binary=:all: \
  -d ./docker_wheels

# 2. Download torch dependencies
pip download --no-deps -d ./docker_wheels \
  filelock typing_extensions sympy networkx jinja2 fsspec "markupsafe" "mpmath==1.3.0"

# 3. Build & run
docker build -t gnn-fraudnet .
docker run -p 8000:8000 gnn-fraudnet
```

Open **http://localhost:8000** → same dashboard, now containerized.

---

## 🌐 Interactive Dashboard

The API serves a **cyberpunk-themed web dashboard** at the root URL (`/`):

| Mode | Description |
|------|-------------|
| **🔍 Single Transaction** | Enter 166 features manually, use presets (zeros, random, suspicious pattern), get instant fraud prediction with visual gauge |
| **📁 CSV Batch Upload** | Drag & drop a CSV file — analyzes all transactions and shows results table with summary stats |
| **🔗 Node Lookup** | Query any node (0–203,768) in the Elliptic graph — uses real graph neighbors for prediction |

**Features:** Ring gauge visualization · Color-coded verdicts (✓ LEGITIMATE / ✗ FRAUDULENT / ? UNCERTAIN) · Batch results table · Live UTC clock · Monospace terminal aesthetic

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive fraud detection dashboard |
| `GET` | `/health` | Health check → `{"status": "ok", "model": "GraphSAGE"}` |
| `POST` | `/predict` | Predict from 166 raw features (JSON body) |
| `GET` | `/predict/{node_id}` | Predict for a known node using real graph neighbors |

### Example: Predict from features

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 1.5, "feature_2": -0.3, ..., "feature_166": 0.0}'
```

### Example: Node lookup

```bash
curl http://localhost:8000/predict/42
```

### Response

```json
{
  "fraud_probability": 0.0753,
  "label": "licit",
  "confidence": "high",
  "node_id": null
}
```

| Probability | Verdict | Action |
|:-----------:|---------|--------|
| < 0.3 | ✅ Licit | Legitimate transaction |
| 0.3 – 0.7 | ⚠️ Uncertain | Needs manual review |
| > 0.7 | 🚨 Illicit | Likely fraudulent |

---

## 📁 Project Structure

```
GNN-FraudNet/
│
├── data/
│   ├── raw/                        ← place the 3 Elliptic CSVs here
│   └── processed/
│       └── elliptic_graph.pt       ← PyG graph (built by notebook 02)
│
├── notebooks/                      ← the executed pipeline (run in order)
│   ├── 01_eda.ipynb                ← dataset exploration
│   ├── 02_graph_construction.ipynb ← build + normalize + mask
│   ├── 03_baseline_models.ipynb    ← LogReg / RF / XGBoost
│   └── 04_gnn_training.ipynb       ← GraphSAGE / GAT + explainability
│
├── src/                            ← reusable importable modules
│   ├── data_loader.py              ← CSVs → PyG Data object
│   ├── graph_builder.py            ← normalize · masks · class weights
│   ├── train.py                    ← training loop + early stopping
│   ├── evaluate.py                 ← F1 · PR-AUC · comparison table
│   ├── explain.py                  ← GNNExplainer + SHAP
│   └── models/
│       ├── graphsage.py            ← GraphSAGE model
│       ├── gat.py                  ← GAT model
│       └── baseline.py             ← LogReg / RF / XGBoost
│
├── api/
│   ├── main.py                     ← FastAPI app + dashboard serving
│   ├── schema.py                   ← Pydantic request/response models
│   └── static/
│       └── index.html              ← interactive web dashboard
│
├── results/
│   ├── metrics.json                ← final metrics (all 5 models)
│   ├── best_graphsage.pt           ← trained model weights
│   ├── best_gat.pt                 ← trained GAT weights
│   └── plots/                      ← EDA, graph, baseline, GNN visualizations
│
├── docker_wheels/                  ← pre-downloaded torch wheels (for Docker build)
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt         ← minimal deps for inference-only container
├── check_transaction.py            ← standalone script to test predictions
├── test_predict.py                 ← API smoke test
└── README.md
```

---

## 🔬 Running the Full Pipeline

The pipeline was built as four notebooks. Run them **in order**:

```bash
jupyter lab notebooks/
```

| # | Notebook | What It Does | Output |
|---|----------|--------------|--------|
| 1 | `01_eda.ipynb` | Explores raw CSVs — label distribution, class imbalance, correlations | `results/plots/eda/` |
| 2 | `02_graph_construction.ipynb` | Builds PyG `Data` object, normalizes features, creates masks | `data/processed/elliptic_graph.pt` |
| 3 | `03_baseline_models.ipynb` | Trains LogReg, Random Forest, XGBoost (no graph) | `results/baseline_metrics.json` |
| 4 | `04_gnn_training.ipynb` | Trains GraphSAGE & GAT, runs GNNExplainer on illicit nodes | `results/best_graphsage.pt` |

> **Tip:** All notebooks have pre-rendered outputs — open them to see results without re-executing.

---

## 🐛 Challenges & Solutions

Building this project involved several non-trivial issues. Here's what we hit and how we solved each:

### 1. Docker Network Timeout

| Problem | `pip install torch==2.4.0` inside Docker timed out repeatedly — the 90 MB wheel kept failing due to Docker's slow virtual network on macOS ARM64 |
|---------|---|
| **Error** | `ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org')` |
| **Solution** | Downloaded the torch wheel locally using `curl -L -C -` (with resume support), then copied it into the Docker image using `COPY docker_wheels/ /tmp/wheels/` and installed with `--no-index --find-links` |

### 2. Dependency Version Conflict (mpmath)

| Problem | `sympy` (required by torch) needs `mpmath<1.4`, but pip downloaded `mpmath==1.4.1` |
|---------|---|
| **Error** | `ERROR: No matching distribution found for mpmath<1.4,>=1.1.0` |
| **Solution** | Manually downloaded `mpmath==1.3.0` into `docker_wheels/` and removed the 1.4.1 version |

### 3. torch_scatter Build Failure

| Problem | `torch_scatter` requires compilation from source and needs torch available during `setup.py`, but pip's isolated build environment doesn't have torch |
|---------|---|
| **Error** | `ModuleNotFoundError: No module named 'torch'` during `Getting requirements to build wheel` |
| **Solution** | Removed `torch_scatter` and `torch_sparse` from Docker requirements — PyG 2.8+ makes them optional. The API works without them |

### 4. sklearn Import Crash in Docker

| Problem | `src/models/__init__.py` imported baseline models (sklearn, xgboost) which aren't installed in the inference-only Docker image |
|---------|---|
| **Error** | `ModuleNotFoundError: No module named 'sklearn'` |
| **Solution** | Wrapped baseline imports in `try/except ImportError` — the API only needs GraphSAGE, not baselines |

### 5. Port Already in Use

| Problem | Starting a second server while one is already running |
|---------|---|
| **Error** | `[Errno 48] Address already in use` |
| **Solution** | Kill the existing process with `lsof -i :8000` → `kill <PID>`, or use a different port: `--port 8001` |

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| GNN Framework | PyTorch Geometric |
| Deep Learning | PyTorch 2.4.0 |
| Baseline Models | XGBoost · Random Forest · scikit-learn |
| Explainability | SHAP · GNNExplainer |
| API | FastAPI · Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (cyberpunk dashboard) |
| Deployment | Docker |
| Visualization | Matplotlib · Seaborn · NetworkX |

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
