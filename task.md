# GNN-FraudNet — Task Tracker

## Phase 0: Environment Setup
- [x] Virtual environment exists (`.venv/`)
- [ ] Verify all imports work

## Phase 1: Data Loading
- [ ] Implement `src/data_loader.py` (4 functions)
- [ ] Test: generate `data/processed/elliptic_graph.pt`

## Phase 2: Graph Preprocessing
- [ ] Implement `src/graph_builder.py` (4 functions)
- [ ] Test: masks + class weights

## Phase 3: EDA Notebooks
- [ ] Fill `notebooks/01_eda.ipynb`
- [ ] Fill `notebooks/02_graph_construction.ipynb`

## Phase 4: Baseline Models
- [ ] Implement `src/models/baseline.py` (4 functions)
- [ ] Fill `notebooks/03_baseline_models.ipynb`

## Phase 5: GNN Models
- [ ] Implement `src/models/graphsage.py`
- [ ] Implement `src/models/gat.py`
- [ ] Update `src/models/__init__.py`

## Phase 6: Training + Evaluation
- [ ] Implement `src/train.py` (4 functions)
- [ ] Implement `src/evaluate.py` (6 functions)
- [ ] Fill `notebooks/04_gnn_training.ipynb`

## Phase 7: Explainability
- [ ] Implement `src/explain.py` (4 functions)
- [ ] Create `notebooks/05_explainability.ipynb`

## Phase 8: API + Docker
- [ ] Implement `api/main.py`
- [ ] Complete `Dockerfile`
