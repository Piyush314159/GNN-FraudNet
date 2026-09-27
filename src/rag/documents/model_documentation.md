# GNN-FraudNet Model Documentation

## Model Architecture: GraphSAGE

The primary fraud detection model is GraphSAGE (Graph SAmple and aggrEGatE), a graph neural network designed for inductive learning on large graphs.

### How GraphSAGE Makes Predictions

1. **Feature Input**: Each transaction node starts with its 166 features.
2. **Message Passing (Layer 1)**: Each node aggregates features from its neighbors using mean pooling. This creates a new representation that combines the node's own features with its neighborhood context.
3. **Transformation**: The aggregated features pass through a learned linear transformation, batch normalization, ReLU activation, and dropout.
4. **Message Passing (Layer 2)**: The process repeats, now incorporating 2-hop neighborhood information. After 2 layers, each node's representation reflects its local graph neighborhood up to 2 hops away.
5. **Classification**: A final linear layer maps the learned representation to 2 output classes (licit vs. illicit).
6. **Probability**: Softmax is applied to get the fraud probability.

### Why GNNs for Fraud Detection

Traditional ML models (XGBoost, Random Forest) treat each transaction independently. They can only use the 166 features of one transaction at a time.

GraphSAGE looks at the transaction AND its neighbors. A transaction surrounded by illicit neighbors is far more likely to be illicit itself — even if its own features look normal. This is the core insight.

However, in the Elliptic dataset, features 94–166 already encode neighborhood statistics. This gives traditional ML models some neighborhood information, which is why XGBoost achieves high performance (0.95 F1). GraphSAGE still demonstrates genuine graph-based learning (0.74 F1, 0.92 PR-AUC).

### Model Performance

| Model | F1 (Illicit) | PR-AUC | Uses Graph? |
|-------|:----------:|:------:|:-----------:|
| Logistic Regression | 0.602 | 0.757 | No |
| Random Forest | 0.950 | 0.983 | No |
| XGBoost | 0.954 | 0.987 | No |
| **GraphSAGE** | **0.739** | **0.917** | **Yes** |
| GAT | 0.449 | 0.615 | Yes |

### Confidence Interpretation

The model outputs a fraud probability between 0 and 1:

- **Below 0.3**: The model is relatively confident the transaction is legitimate.
- **0.3 to 0.7**: The model is uncertain. Manual review is recommended.
- **Above 0.7**: The model is relatively confident the transaction is illicit.

Confidence is derived from how far the probability is from the 0.5 decision boundary. A probability of 0.95 indicates much higher confidence than 0.55.

### Model Limitations

1. **Static graph**: The model was trained on a fixed snapshot of the Bitcoin transaction graph. It cannot incorporate truly new, unseen transactions without retraining or graph extension.
2. **Label scarcity**: Only 23% of nodes have labels (4,545 illicit, 42,019 licit, 157,205 unknown). The model's understanding is limited by available ground truth.
3. **Class imbalance**: Illicit transactions are ~9x rarer than licit ones. The model uses class-weighted loss to handle this, but rare fraud patterns may still be underrepresented.
4. **Anonymized features**: The 166 features are anonymized, limiting interpretability.
5. **2-hop receptive field**: The model only sees 2 hops of neighborhood. Fraud patterns spanning larger graph distances may be missed.
6. **No temporal modeling**: Although time step is a feature, the model doesn't explicitly model temporal sequences of transactions.

## Explainability Methods

### GNNExplainer

GNNExplainer identifies which edges and features were most important for a specific node's prediction:

- **Edge importance**: Which connections to neighbors mattered most for the prediction.
- **Feature importance**: Which of the 166 features had the most influence.
- **Node-specific**: Each explanation is for one specific prediction, not a global explanation.
- **Limitation**: GNNExplainer optimizes a local approximation and may not capture all relevant factors.

### SHAP (SHapley Additive exPlanations)

SHAP provides a game-theoretic approach to feature importance:

- **Additive**: SHAP values for all features sum to the difference between the prediction and the baseline.
- **Consistent**: Features with larger impact always get larger SHAP values.
- **Computationally expensive**: We use KernelExplainer, which requires multiple forward passes.
- **Per-node**: Each node gets its own set of SHAP values.
- **Limitation**: SHAP perturbs only the target node's features while holding neighbors fixed. It may underestimate the importance of features that interact with neighbor features through message passing.
