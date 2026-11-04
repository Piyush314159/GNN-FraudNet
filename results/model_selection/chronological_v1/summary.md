# First chronological comparison

Cleanup status: the superseded XGBoost checkpoint and retired comparison code
were removed after the second round selected Random Forest. The metrics below
record the completed first experiment; they do not describe the currently
retained artifact. See the [current workflow](../../../docs/random-forest.md).

This run is an exploratory, single-seed screen. It does not change the production API.

## Validation results

| Model | AP | Fraud F1 | Decision |
|---|---:|---:|---|
| graphsage | 0.3754 | 0.4093 | reference |
| xgboost | 0.9252 | 0.8926 | keep |
| random_forest | 0.9264 | 0.9052 | reject |
| mlp | 0.8818 | 0.8622 | reject |
| resnet | 0.8761 | 0.8598 | reject |
| transformer | 0.8803 | 0.8597 | reject |
| graphsage_xgboost | 0.9234 | 0.8934 | reject |

Random Forest improved AP by only 0.0012 over XGBoost, below the predeclared 0.002 minimum; it had higher validation F1. Its rejection does not establish that it is inferior.

## Final held-out comparison

| Model | AP | Fraud F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Fresh GraphSAGE | 0.1853 | 0.2483 | 0.2055 | 0.3137 |
| Selected XGBoost | 0.5307 | 0.5674 | 0.7870 | 0.4436 |

XGBoost passed the final improvement gate and was the only retained new model artifact from this round. Rejected weights were never serialized. At experiment time, its saved model, scaler, and validation-selected threshold exactly reproduced the test metrics after reloading. Those superseded files were subsequently removed during cleanup.

The test confusion matrix is 8,384 true negatives, 49 false positives, 227 false negatives, and 181 true positives. At the frozen threshold, 55.6% of the 408 illicit transactions are missed. AP falls from 0.9252 on validation to 0.5307 on test. This time-period degradation requires further chronological robustness work; these results do not establish deployment readiness.

See the [experiment protocol](../../../docs/model-selection.md) for the feature contract, split protocol, selection rule, and limitations. Historical random-split scores are not comparable. The final test set is now observed and must not be reused for further model selection.

The active GraphSAGE API and historical checkpoints are unchanged. Integrating XGBoost requires its matching preprocessing, threshold, and SHAP explanations rather than the existing GNN explanation workflow.
