# Elliptic Dataset Feature Definitions

## Overview

The Elliptic Bitcoin dataset contains 166 features per transaction node. These features are divided into two groups:

- **Features 1–93**: Local transaction features (properties of the transaction itself).
- **Features 94–166**: Aggregated neighborhood features (statistics computed over the transaction's 1-hop neighbors).

The exact feature names are anonymized by the dataset creators for privacy, but their categories are documented.

## Local Transaction Features (1–93)

These describe the transaction in isolation:

- **Transaction amount features**: Total input value, total output value, transaction fee, number of inputs, number of outputs.
- **Timing features**: Time step (1–49), timestamp-derived features, time since last transaction from same address.
- **Flow features**: Ratio of input to output value, change amount, proportion of inputs from known addresses.
- **Structural features**: Number of unique input addresses, number of unique output addresses, whether it is a coinbase transaction.

When SHAP identifies a local feature as important, it means the transaction's own properties (not its neighbors) are driving the prediction.

## Aggregated Neighborhood Features (94–166)

These are statistics computed over 1-hop neighbors:

- **Mean, standard deviation, min, max** of neighbor transaction features.
- **Neighbor count statistics**: Number of neighbors, number of inputs/outputs across neighbors.
- **Neighbor label statistics**: Although labels are not directly used as features, temporal patterns in neighbor behavior create indirect label signals.

When SHAP identifies an aggregated feature (94–166) as important, it means the node's neighborhood context is driving the prediction. This is where the GNN adds value over traditional ML.

## Feature Interpretation Guide

### High-Importance Local Features (1–93)

If a local feature has high SHAP importance:
- The transaction itself has unusual properties.
- Check: Is the transaction amount abnormal? Is the timing suspicious?
- This indicates the transaction would be flagged even without graph context.

### High-Importance Aggregated Features (94–166)

If an aggregated feature has high SHAP importance:
- The transaction's neighborhood is unusual.
- Check: Are neighbors illicit? Is the neighborhood structure abnormal?
- This indicates the graph context is contributing to the fraud signal.
- This is the GNN's primary advantage over traditional ML models.

### Feature Direction

- **Positive SHAP value**: The feature increases the fraud probability.
- **Negative SHAP value**: The feature decreases the fraud probability (i.e., it looks legitimate).
- **Large absolute SHAP value**: The feature has a strong influence regardless of direction.

## Limitations of Anonymous Features

Because the 166 features are anonymized:
- We cannot map features to specific real-world transaction properties.
- SHAP provides relative importance, but interpretations must be approximate.
- Feature names like "feature_14" are proxies — we know the feature matters, but not exactly what it measures.
- The local vs. aggregated distinction (1–93 vs. 94–166) is the most reliable interpretation available.
