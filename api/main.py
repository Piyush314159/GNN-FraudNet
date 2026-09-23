"""
main.py
=======
PURPOSE:
    FastAPI application exposing the trained GNN model as a REST API.
    Allows any system to send transaction features and get a fraud score back.

ENDPOINTS:
    GET  /health              → returns {"status": "ok", "model": "GraphSAGE"}
    POST /predict             → takes TransactionRequest, returns FraudResponse
    GET  /predict/{node_id}   → predict for a known node ID in the Elliptic graph

HOW IT WORKS:
    startup_event(app)
        Runs when app starts
        Load trained GraphSAGE model from results/best_graphsage.pt
        Load PyG Data object from data/processed/elliptic_graph.pt
        Store both in app state

    health()
        Return model name and status
        Useful for Docker health checks

    predict(request: TransactionRequest)
        Convert 166 input floats → tensor (isolated node, no edges)
        Run model forward pass
        Apply softmax → get illicit probability
        Return FraudResponse with probability + label

    predict_by_node(node_id: int)
        Look up node_id in loaded graph
        Run model on the full graph (so message passing uses real neighbors)
        Return FraudResponse

HOW TO RUN:
    uvicorn api.main:app --reload --port 8000
"""
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from api.schema import FraudResponse, TransactionRequest
from src.models.graphsage import GraphSAGE

MODEL_PATH = os.getenv("MODEL_PATH", "results/best_graphsage.pt")
DATA_PATH = os.getenv("DATA_PATH", "data/processed/elliptic_graph.pt")
NUM_FEATURES = 166
HIDDEN_CHANNELS = 64


def _confidence_bucket(prob):
    margin = abs(prob - 0.5)
    if margin >= 0.3:
        return "high"
    if margin >= 0.1:
        return "medium"
    return "low"


def startup_event(app: FastAPI):
    model = GraphSAGE(in_channels=NUM_FEATURES, hidden_channels=HIDDEN_CHANNELS, out_channels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)

    app.state.model = model
    app.state.data = data


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_event(app)
    yield


app = FastAPI(title="GNN-FraudNet", description="GNN-based fraud detection API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "GraphSAGE"}


@app.post("/predict", response_model=FraudResponse)
def predict(request: TransactionRequest):
    features = [getattr(request, f"feature_{i}") for i in range(1, NUM_FEATURES + 1)]
    x = torch.tensor([features], dtype=torch.float)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    with torch.no_grad():
        logits = app.state.model(x, edge_index)
        probs = torch.softmax(logits, dim=1)[0]

    fraud_prob = probs[1].item()
    return FraudResponse(
        fraud_probability=fraud_prob,
        label="illicit" if fraud_prob >= 0.5 else "licit",
        confidence=_confidence_bucket(fraud_prob),
    )


@app.get("/predict/{node_id}", response_model=FraudResponse)
def predict_by_node(node_id: int):
    data = app.state.data
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    with torch.no_grad():
        logits = app.state.model(data.x, data.edge_index)
        probs = torch.softmax(logits[node_id], dim=0)

    fraud_prob = probs[1].item()
    return FraudResponse(
        fraud_probability=fraud_prob,
        label="illicit" if fraud_prob >= 0.5 else "licit",
        confidence=_confidence_bucket(fraud_prob),
        node_id=node_id,
    )
