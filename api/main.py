"""
main.py
=======
PURPOSE:
    FastAPI application exposing the trained GNN model as a REST API.
    Allows any system to send transaction features and get a fraud score back.

ENDPOINTS:
    GET  /health         → returns {"status": "ok", "model": "GraphSAGE"}
    POST /predict        → takes TransactionRequest, returns FraudResponse
    GET  /predict/{node_id} → predict for a known node ID in the Elliptic graph

HOW IT WORKS:
    startup_event()
        Runs when app starts
        Load trained GraphSAGE model from results/best_model.pt
        Load PyG Data object from data/processed/elliptic_graph.pt
        Store both in app state

    health()
        Return model name and status
        Useful for Docker health checks

    predict(request: TransactionRequest)
        Convert 166 input floats → tensor
        Run model forward pass
        Apply softmax → get illicit probability
        Return FraudResponse with probability + label

    predict_by_node(node_id: int)
        Look up node_id in loaded graph
        Run model on that node's features + neighborhood
        Return FraudResponse

HOW TO RUN:
    uvicorn api.main:app --reload --port 8000

TODO: Implement FastAPI app below
"""

# from fastapi import FastAPI
# from api.schema import TransactionRequest, FraudResponse

# app = FastAPI(title="GNN-FraudNet", description="GNN-based fraud detection API")

def startup_event():
    # TODO: load model and data on startup
    pass

def health():
    # TODO: return status dict
    pass

def predict(request):
    # TODO: run model inference, return FraudResponse
    pass

def predict_by_node(node_id):
    # TODO: lookup node, run inference, return FraudResponse
    pass
