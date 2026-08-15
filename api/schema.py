from typing import Optional
from pydantic import BaseModel, Field, create_model


# Dynamically build TransactionRequest with feature_1 .. feature_166
_fields = {f"feature_{i}": (float, Field(...)) for i in range(1, 167)}
TransactionRequest = create_model("TransactionRequest", **_fields)


class FraudResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    label: str  # "illicit" or "licit"
    confidence: str  # "high", "medium", or "low"
    node_id: Optional[int] = None
