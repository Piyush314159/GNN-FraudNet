# Chronological model selection

This is the historical protocol for the completed first comparison. Its retired
experiment runner and rejected model implementations were removed during cleanup.
Metrics, dataset fingerprints and the summary remain under
`results/model_selection/chronological_v1/`. The superseded XGBoost checkpoint
was removed after Random Forest became the retained research candidate.

Use the [current Random Forest workflow](random-forest.md) to train the retained
recipe. This document records the earlier experiment rather than a runnable sweep.

## Protocol fixed before training

- Training: periods 1–34; validation: 35–41; final test: 42–49.
- Only labelled transactions contribute supervised examples. Unknown nodes can
  provide graph context, but their labels are never training targets.
- All models use the same 93 local transaction attributes: raw CSV columns 2–94
  (zero-based). The timestep and the 72 supplied aggregate attributes are excluded.
  In a 166-feature API vector, this corresponds to `features[1:94]`.
- Standardization is fitted on labelled training transactions only.
- GraphSAGE training sees graph nodes/edges through period 34. Validation and test
  predictions are made separately for each period using a graph snapshot ending
  at that period. Future nodes and edges are excluded.
- Hybrid training embeddings come from the period-34 training snapshot;
  validation/test embeddings come from the corresponding period's snapshot.
- The baseline is a freshly trained GraphSAGE, rather than the historical
  checkpoint trained with a random split and different input features.
- Candidate order: XGBoost, Random Forest, MLP, tabular residual network, small
  feature-token Transformer, then GraphSAGE embeddings + XGBoost.
- Each model's binary decision threshold maximizes F1 on validation predictions.
  Neural early stopping selects weights by validation average precision (AP).
- A challenger replaces the incumbent only if validation AP increases by at
  least 0.002 and validation fraud F1 does not decrease. This is an engineering
  screening threshold, not a statistical significance test.
- After candidate selection is frozen, evaluate the selected model and the
  GraphSAGE reference once on the test set, using their validation thresholds.
  Retain the candidate artifact only if it also passes the same improvement rule
  against the reference on this final comparison. A failed final check does not
  trigger testing of runners-up.

## Outputs and retention

`comparison.json` records validation metrics, decisions, timings, dataset hashes,
split sizes, and the final test comparison. Here, the `pr_auc` field is sklearn's
average precision, consistent with the existing project metric convention.

The original run serialized the selected XGBoost candidate with its preprocessing
and metadata. Those superseded files were removed during cleanup. Rejected
candidate weights were never retained. Metrics and recorded settings remain
available for audit; the retired source fingerprints are historical records.
The current Random Forest checkpoint and the GraphSAGE checkpoint used by the
API remain available.

The API is not automatically switched. Its current explanation workflow uses
GraphSAGE; changing only its classifier would incorrectly attach GNN explanations
to another model's predictions. A serving integration must use the selected
model's preprocessing, threshold, and matching explanation method together.

## Limits

This is a single-seed comparison with fixed, bounded model configurations, not a
comprehensive architecture search. The Transformer is a small feature-token
implementation, not a reproduction of the published FT-Transformer benchmark.
Validation is reused for early stopping, threshold selection, and model choice.
The held-out test cannot remain an untouched benchmark for future rounds after
its scores have been reviewed.

The data provides period-level timestamps, not within-period event ordering or
feature publication times. Graph context within a period is available to this
offline benchmark. Local attributes also come precomputed from the dataset;
their event-time availability has not been independently verified. Therefore,
these results do not establish live transaction performance. They cannot be
compared directly with the historical README scores.

Before deployment, validate stability across seeds and additional chronological
windows, measure calibration and latency, and integrate consistent labels and
model-appropriate explanations throughout the API and dashboard.
