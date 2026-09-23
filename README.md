# GNN-FraudNet

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch_Geometric-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/status-results_available-brightgreen?style=flat"/>
</p>

<p align="center">
  <b>Graph Neural Network based fraud detection on Bitcoin transaction graphs.</b><br/>
  GraphSAGE · GAT · SHAP + GNNExplainer · REST API
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
| Illicit (fraud) | 4,545 |
| Licit (legit) | 42,019 |
| Unknown | 157,205 |

> Download → [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)  
> Place the 3 CSVs in `data/raw/`

---

## Results

Test-set metrics from a chronological-agnostic 70/15/15 split of the labeled nodes (see `results/metrics.json`):

| Model | F1 (Illicit) | PR-AUC | Accuracy | Uses Graph? |
|-------|:-----------:|:------:|:--------:|:-----------:|
| Logistic Regression | 0.602 | 0.757 | 0.880 | ✗ |
| Random Forest | 0.950 | 0.983 | 0.991 | ✗ |
| **XGBoost** | **0.954** | **0.987** | **0.991** | ✗ |
| GraphSAGE | 0.739 | 0.917 | 0.936 | ✓ |
| GAT | 0.449 | 0.615 | 0.779 | ✓ |

**Honest takeaway:** on this dataset, tree-based baselines (XGBoost, Random Forest) beat both GNNs. The 166 engineered features already encode a lot of local + aggregated neighborhood signal, so graph structure adds less than the "GNNs always win" narrative suggests. GraphSAGE clearly beats a linear model and beats GAT — GAT likely needs more tuning (attention dropout, heads, LR schedule) to close the gap. See `notebooks/04_gnn_training.ipynb` §15 for the full discussion.

---

## Project Structure

```
GNN-FraudNet/
│
├── data/
│   ├── raw/                        ← place the 3 Elliptic CSVs here
│   └── processed/
│       └── elliptic_graph.pt       ← PyG graph (built by notebook 02 / data_loader.py)
│
├── notebooks/                      ← the actual, executed pipeline
│   ├── 01_eda.ipynb                ← dataset exploration       → results/plots/eda/
│   ├── 02_graph_construction.ipynb ← build + normalize + mask  → results/plots/graph/
│   ├── 03_baseline_models.ipynb    ← LogReg / RF / XGBoost     → results/plots/baseline/
│   └── 04_gnn_training.ipynb       ← GraphSAGE / GAT + explain → results/plots/gnn/
│
├── src/                            ← reusable version of the same logic
│   ├── data_loader.py              ← CSVs → PyG Data object
│   ├── graph_builder.py            ← normalize · masks · class weights
│   ├── train.py                    ← training loop + early stopping
│   ├── evaluate.py                 ← F1 · PR-AUC · comparison table
│   ├── explain.py                  ← GNNExplainer + SHAP
│   └── models/
│       ├── graphsage.py
│       ├── gat.py
│       └── baseline.py
│
├── api/
│   ├── main.py                     ← FastAPI endpoints
│   └── schema.py                   ← request/response models
│
├── results/
│   ├── metrics.json                ← final metrics, all 5 models
│   ├── baseline_metrics.json       ← LogReg / RF / XGBoost only
│   ├── best_graphsage.pt           ← trained GraphSAGE weights (loaded by the API)
│   ├── best_gat.pt                 ← trained GAT weights
│   └── plots/
│       ├── eda/                    ← 11 charts from notebook 01
│       ├── graph/                  ← degree distribution from notebook 02
│       ├── baseline/               ← confusion matrices, PR curves, feature importance
│       └── gnn/                    ← training curves, eval plots, GNNExplainer subgraphs
│
├── explainer/                      ← project docs (roadmap, task tracker)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## How to Run

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

Download the [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place the 3 CSVs in `data/raw/`:

```
data/raw/elliptic_txs_features.csv
data/raw/elliptic_txs_edgelist.csv
data/raw/elliptic_txs_classes.csv
```

### 3. Run the pipeline

The pipeline was built and validated as four notebooks, run **in order**. Each one reads the previous notebook's output and writes its own results/plots:

```bash
jupyter lab notebooks/
```

| Notebook | What it does | Produces |
|----------|--------------|----------|
| `01_eda.ipynb` | Explores the raw CSVs — label distribution, class imbalance, correlations, degree distribution, temporal fraud rate | `results/plots/eda/*.png` |
| `02_graph_construction.ipynb` | Builds the PyG `Data` object, normalizes features (train-stats only), creates train/val/test masks, computes class weights | `data/processed/elliptic_graph.pt`, `results/plots/graph/*.png` |
| `03_baseline_models.ipynb` | Trains Logistic Regression, Random Forest, XGBoost on features only (no graph) | `results/baseline_metrics.json`, `results/plots/baseline/*.png` |
| `04_gnn_training.ipynb` | Trains GraphSAGE and GAT, evaluates all 5 models side by side, runs GNNExplainer on sample illicit nodes | `results/best_graphsage.pt`, `results/best_gat.pt`, `results/metrics.json`, `results/plots/gnn/*.png` |

Run each notebook top-to-bottom ("Run All"). Each cell prints its own metrics/tables and renders its own plots inline as it runs.

### 4. (Optional) Run the same steps as a script instead of notebooks

`src/` is a reusable, importable version of the exact same logic used in the notebooks:

```python
from src.data_loader import build_pyg_data
from src.graph_builder import create_masks, normalize_features, get_class_weights, inspect_graph
from src.models.graphsage import GraphSAGE
from src.train import train

data = build_pyg_data("data/raw", "data/processed")   # → data/processed/elliptic_graph.pt
data = create_masks(data)
data = normalize_features(data)
weights = get_class_weights(data)
inspect_graph(data)

model = GraphSAGE(in_channels=166, hidden_channels=64, out_channels=2)
model, history = train(
    model, data,
    config={"lr": 0.001, "epochs": 200, "patience": 20, "weight_decay": 5e-4},
    class_weights=weights,
    model_name="GraphSAGE",
)
```

### 5. Serve the trained model via the API

```bash
uvicorn api.main:app --reload --port 8000
```

This loads `results/best_graphsage.pt` + `data/processed/elliptic_graph.pt` on startup (override paths with the `MODEL_PATH` / `DATA_PATH` env vars).

### 6. Run with Docker instead

```bash
docker build -t gnn-fraudnet .
docker run -p 8000:8000 gnn-fraudnet
```

---

## How to See the Results

- **Fastest:** open any notebook in `notebooks/` — every metric, table, and chart is already rendered inline from the last run, no need to re-execute anything.
- **Plots on disk:** browse `results/plots/{eda,graph,baseline,gnn}/` — PNGs for label distribution, correlations, degree distributions, PR curves, confusion matrices, XGBoost feature importance, GraphSAGE/GAT training curves, and GNNExplainer subgraph visualizations for 3 sample illicit nodes.
- **Numbers:** `results/metrics.json` (all 5 models) and `results/baseline_metrics.json` (baselines only) — F1 / PR-AUC / accuracy per model.
- **Live predictions:** start the API (step 5 above) and hit it:

```bash
# Health check
curl http://localhost:8000/health

# Predict for an existing node in the graph (uses its real neighbors)
curl http://localhost:8000/predict/5

# Predict from raw 166 features (treated as an isolated node, no graph context)
python3 -c "
import json, urllib.request
payload = {f'feature_{i}': 0.0 for i in range(1, 167)}
req = urllib.request.Request(
    'http://localhost:8000/predict',
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'},
)
print(urllib.request.urlopen(req).read().decode())
"
```

**Response**
```json
{
  "fraud_probability": 0.0068,
  "label": "licit",
  "confidence": "high",
  "node_id": 5
}
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| GNN Framework | PyTorch Geometric |
| Baseline Models | XGBoost · Random Forest · scikit-learn |
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
