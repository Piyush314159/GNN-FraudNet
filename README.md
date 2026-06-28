# GNN-FraudNet

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch_Geometric-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/status-in_progress-yellow?style=flat"/>
</p>

<p align="center">
  <b>Graph Neural Network based fraud detection on Bitcoin transaction graphs.</b><br/>
  GraphSAGE · GAT · SHAP Explainability · REST API
</p>

---

## Why Graph?

Traditional ML models treat each transaction in isolation.  
**GNN-FraudNet exploits the graph** — a transaction surrounded by illicit nodes is far more likely to be illicit.

```
 [Account A] ──→ [Transaction X] ──→ [Account B]
                       │
               [FLAGGED: illicit neighbors]
```

The same message-passing framework used in particle physics (Belle II experiment)  
applies naturally to financial transaction graphs.

---

## Architecture

```
Elliptic Bitcoin Dataset (203k nodes · 234k edges)
              ↓
   Node Features (166) + Edge Index
              ↓
   Normalize · Mask · Class Weights
              ↓
   ┌──────────────────────────────┐
   │  GraphSAGE  ·  GAT          │
   │  2-layer GNN → node logits  │
   └──────────────────────────────┘
              ↓
   Fraud Probability per Transaction
              ↓
   SHAP + GNNExplainer  ←  Why was this flagged?
              ↓
   FastAPI  ·  Docker
```

---

## Dataset

**Elliptic Bitcoin Dataset** — one of the few real-world labeled cryptocurrency fraud datasets.

| Property | Value |
|----------|-------|
| Nodes (transactions) | 203,769 |
| Edges (BTC flows) | 234,355 |
| Features per node | 166 |
| Illicit (fraud) | ~4,545 |
| Licit (legit) | ~42,019 |
| Unknown | ~157,205 |

> Download → [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)  
> Place files in `data/raw/`

---

## Results

| Model | F1 (Illicit) | PR-AUC | Uses Graph? |
|-------|:-----------:|:------:|:-----------:|
| Logistic Regression | — | — | ✗ |
| XGBoost | — | — | ✗ |
| GraphSAGE | — | — | ✓ |
| GAT | — | — | ✓ |

> Results will be filled after training is complete.

---

## Project Structure

```
GNN-FraudNet/
│
├── data/
│   ├── raw/                    ← place Elliptic CSVs here
│   └── processed/              ← PyG graph after preprocessing
│
├── notebooks/
│   ├── 01_eda.ipynb            ← dataset exploration
│   ├── 02_graph_construction.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_gnn_training.ipynb
│
├── src/
│   ├── data_loader.py          ← CSVs → PyG Data object
│   ├── graph_builder.py        ← normalize · masks · class weights
│   ├── train.py                ← training loop + early stopping
│   ├── evaluate.py             ← F1 · PR-AUC · comparison table
│   ├── explain.py              ← GNNExplainer + SHAP
│   └── models/
│       ├── graphsage.py
│       ├── gat.py
│       └── baseline.py
│
├── api/
│   ├── main.py                 ← FastAPI endpoints
│   └── schema.py               ← request/response models
│
├── results/
│   ├── metrics.json
│   └── plots/                  ← PR curves · confusion matrix · SHAP
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset → place in data/raw/

# 3. Build graph object
python -c "from src.data_loader import build_pyg_data; build_pyg_data('data/raw', 'data/processed')"

# 4. Train
python src/train.py

# 5. Start API
uvicorn api.main:app --reload --port 8000
```

---

## API

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 0.5, "feature_2": 1.2, ...}'
```

**Response**
```json
{
  "fraud_probability": 0.91,
  "label": "illicit",
  "confidence": "high"
}
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| GNN Framework | PyTorch Geometric |
| Baseline Models | XGBoost · scikit-learn |
| Explainability | SHAP · GNNExplainer |
| API | FastAPI · Uvicorn |
| Deployment | Docker |
| Visualization | Matplotlib · Seaborn · NetworkX |

---

## Background

This project bridges **particle physics and fintech**.  
The GNN message-passing framework applied here originates from work on the  
[Belle II experiment](https://www.belle2.org/) — where GNNs reconstruct  
particle decay trees from detector hits.  
The same structural reasoning applies to financial transaction graphs.

---

<p align="center">
  Made by <a href="https://piyush314159.github.io">Piyush</a> · MSc Physics · IIT Hyderabad
</p>