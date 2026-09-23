"""
Check a new Bitcoin transaction for fraud using GNN-FraudNet.

Usage:
    1. Start the server:  docker run -p 8000:8000 gnn-fraudnet
    2. Run this script:   python check_transaction.py
"""

import json
import urllib.request

API_URL = "http://localhost:8000/predict"


def check_transaction(features: dict) -> dict:
    """Send a transaction to the API and get fraud prediction."""
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(features).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())


# ──────────────────────────────────────────────────────────────
# Example 1: A normal-looking transaction (low feature values)
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 1: Normal-looking transaction")
print("=" * 60)

normal_tx = {f"feature_{i}": 0.0 for i in range(1, 167)}
result = check_transaction(normal_tx)

print(f"  Fraud probability: {result['fraud_probability']:.1%}")
print(f"  Label:             {result['label']}")
print(f"  Confidence:        {result['confidence']}")
print()

# ──────────────────────────────────────────────────────────────
# Example 2: A suspicious transaction (high feature values)
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 2: Suspicious transaction")
print("=" * 60)

suspicious_tx = {f"feature_{i}": 2.5 for i in range(1, 167)}
result = check_transaction(suspicious_tx)

print(f"  Fraud probability: {result['fraud_probability']:.1%}")
print(f"  Label:             {result['label']}")
print(f"  Confidence:        {result['confidence']}")
print()

# ──────────────────────────────────────────────────────────────
# Example 3: Your own custom transaction
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 3: Custom transaction (mix of values)")
print("=" * 60)

custom_tx = {f"feature_{i}": 0.0 for i in range(1, 167)}
# Set some features to unusual values
custom_tx["feature_1"] = 1.5    # e.g., transaction amount
custom_tx["feature_2"] = -0.8   # e.g., time since last tx
custom_tx["feature_10"] = 3.2   # e.g., number of inputs
custom_tx["feature_50"] = -2.1  # e.g., flow pattern

result = check_transaction(custom_tx)

print(f"  Fraud probability: {result['fraud_probability']:.1%}")
print(f"  Label:             {result['label']}")
print(f"  Confidence:        {result['confidence']}")
print()

# ──────────────────────────────────────────────────────────────
# How to interpret results
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("How to interpret results:")
print("=" * 60)
print("  fraud_probability < 0.3  → licit  (high confidence)")
print("  fraud_probability 0.3-0.7 → uncertain (low confidence)")
print("  fraud_probability > 0.7  → illicit (high confidence)")
