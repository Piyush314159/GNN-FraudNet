"""
schema.py
=========
PURPOSE:
    Define Pydantic models for FastAPI request and response validation.

HOW IT WORKS:
    TransactionRequest
        166 float fields representing Elliptic node features
        Field names: feature_1, feature_2, ..., feature_166
        FastAPI will auto-validate types and reject malformed requests

    FraudResponse
        fraud_probability : float between 0 and 1
        label             : string — "illicit" if prob > 0.5 else "licit"
        confidence        : string — "high" / "medium" / "low" based on prob distance from 0.5
        node_id           : optional int if querying by node ID

TODO: Implement Pydantic models below
"""

# from pydantic import BaseModel

class TransactionRequest:
    # TODO: 166 float fields using Pydantic BaseModel
    pass

class FraudResponse:
    # TODO: fraud_probability, label, confidence fields
    pass
