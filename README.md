# GNN-FraudNet

> Graph Neural Network based fraud detection on Bitcoin transaction graphs.

---

## Overview

GNN-FraudNet detects illicit (fraudulent) transactions in a Bitcoin transaction graph
using Graph Neural Networks (GraphSAGE and GAT) implemented in PyTorch Geometric.

Unlike traditional ML models that treat each transaction independently,
GNN-FraudNet exploits the graph structure — a transaction surrounded by illicit
nodes is itself more likely to be illicit.

---

## Architecture

```
Raw Transaction Graph (Elliptic Dataset)
        ↓
  Node Features (166) + Edge Index
        ↓
  Feature Normalization + Train/Val/Test Masks
        ↓
  ┌─────────────────────────────────┐
  │  GraphSAGE / GAT               │
  │  2-layer GNN → node logits     │
  └─────────────────────────────────┘
        ↓
  Fraud Probability per Transaction
        ↓
  SHAP + GNNExplainer (Why was it flagged?)
        ↓
  FastAPI REST Endpoint
```

---

## Dataset

**Elliptic Bitcoin Dataset** (Kaggle)
- 203,769 transaction nodes
- 234,355 directed edges
- 166 features per node (transaction metadata)
- Labels: illicit (fraud) / licit (legit) / unknown

Download from: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Place files in `data/raw/`

---

## Project Structure

```
GNN-FraudNet/
├── data/raw/               ← place Elliptic CSVs here
├── data/processed/         ← PyG graph saved here after preprocessing
├── notebooks/              ← EDA, graph construction, training notebooks
├── src/
│   ├── data_loader.py      ← load CSVs → PyG Data object
│   ├── graph_builder.py    ← normalize, mask, class weights
│   ├── models/
│   │   ├── graphsage.py    ← GraphSAGE model
│   │   ├── gat.py          ← GAT model
│   │   └── baseline.py     ← XGBoost + LogReg
│   ├── train.py            ← training loop
│   ├── evaluate.py         ← metrics + comparison table
│   └── explain.py          ← GNNExplainer + SHAP
├── api/
│   ├── main.py             ← FastAPI app
│   └── schema.py           ← request/response models
├── results/
│   ├── metrics.json        ← filled after training
│   └── plots/              ← PR curves, confusion matrices, SHAP plots
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Results

| Model | F1 (Illicit) | PR-AUC |
|-------|-------------|--------|
| Logistic Regression | TODO | TODO |
| XGBoost | TODO | TODO |
| GraphSAGE | TODO | TODO |
| GAT | TODO | TODO |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Elliptic dataset and place in data/raw/

# 3. Build graph
python -c "from src.data_loader import build_pyg_data; build_pyg_data('data/raw', 'data/processed')"

# 4. Train models
python src/train.py

# 5. Start API
uvicorn api.main:app --reload --port 8000
```

---

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict fraud probability
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 0.5, "feature_2": 1.2, ...}'
```

---

## Tech Stack

- PyTorch Geometric — GNN implementation
- XGBoost — baseline model
- SHAP + GNNExplainer — explainability
- FastAPI — REST API
- Docker — deployment

---

## Background

This project extends GNN research from particle physics (Belle II experiment)
to financial fraud detection. The same message-passing framework used to
reconstruct particle decay trees applies naturally to transaction graphs.
