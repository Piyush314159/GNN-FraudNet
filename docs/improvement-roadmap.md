# FraudNet improvement priorities

Agreed direction: compare models sequentially under a fixed protocol, retain
improving candidates, and discard unsuccessful trained artifacts. Preserve
experiment metrics and recipes for auditability.

1. **Consistent predictions:** centralize preprocessing and classification
   thresholds across prediction, investigation, and dashboard responses.
2. **Trustworthy explanations:** distinguish unsigned GNN importance from
   direction, use the correct raw-logit explainer configuration, and remove
   unsupported graph-versus-feature interpretations.
3. **Reliable reports:** align provider support with documentation, handle empty
   evidence, and clearly identify fallback reports and partial failures.
4. **Reproducible data preparation:** reconcile the headerless CSV loader with
   notebooks and retain preprocessing metadata with model weights.
5. **Credible evaluation:** chronological splits, validation-only selection,
   additional time windows/seeds, calibrated scores, and matched baseline inputs.
6. **Deployment/dashboard reliability:** meaningful build checks, consistent
   network configuration, sanitized report rendering, and stream cancellation.

## First model comparison completed

See [the experiment protocol](model-selection.md) and
[the recorded comparison](../results/model_selection/chronological_v1/summary.md).
XGBoost passed the screening rule and final comparison with the fresh GraphSAGE
reference. MLP, residual network, small feature-token Transformer, and the graph
hybrid did not improve on XGBoost under the rule. Random Forest's AP improvement
was smaller than the required minimum, despite its higher validation F1.

Only XGBoost's new trained artifact was retained. The production API and
historical models remain unchanged. Strong degradation in later periods means
that robustness and threshold validation remain priorities before activation.
The observed final test set must not become a tuning set for follow-up rounds.

## Second comparison completed

The [rolling comparison](../results/model_selection/rolling_v2/summary.md)
excludes the observed final periods and separates inner tuning from evaluation.
Random Forest now leads this development comparison, averaging AP 0.8902 and
fraud F1 0.8616 against XGBoost's 0.8862 and 0.8469 over three windows/two seeds.
Its advantage is modest and depends on the period; it was weaker in window one.

The researchers' official MLP and ResNet, and the GraphSAGE + XGBoost hybrid,
did not improve sufficiently. The official FT-Transformer was stopped after
three completed runs because even perfect missing results could not satisfy
the unchanged two-window consistency criterion. Its numerical rejection proof
is recorded; no overall three-window average was invented.

Only the new Random Forest research checkpoint is retained from this round.
Historical round-one artifacts remain as evidence. The API is still unchanged;
consistent prediction semantics and model-appropriate explanations remain the
next integration priorities. See the [protocol](rolling-comparison.md) for limits.

## Cleanup completed

Retired experiment runners, unused standalone GAT/baseline/neural-model code,
superseded XGBoost weights, GAT weights/individual plots, duplicate environments
and old API demo scripts were removed at the user's request. Their dedicated
tests and optional research dependency were also removed. The selected Random
Forest checkpoint, comparison metrics, notebooks and authored reports remain.

The [Random Forest workflow](random-forest.md) is the current research training
entrypoint. The simpler trainer reproduces the retained real-data predictions
and evaluation metrics. The active GraphSAGE model, its training utilities and
the API's explanation dependencies remain until a complete serving migration.
Historical notebook implementations and report listings are research records,
not the current training interface. See the [cleanup manifest](../results/model_selection/cleanup.json).

## Dashboard redesign completed

The [dashboard](dashboard.md) now uses a responsive analyst workspace with
graph-node/feature input, network evidence, on-demand reports, session history,
validated CSV screening and exports, and actual dataset/service information.
Remote fonts and runtime CDNs were removed. Report content and user-controlled
labels are rendered as text, and streamed reports support cancellation/retry.
Frontend classifications now follow the server's returned label.

Local browser checks covered real predictions and local-summary reports, batch
upload/export, validation, mobile layouts, cancellation and disconnection states.
The serving model remains GraphSAGE. Unified backend classification thresholds,
report/explainer correctness and migration to the retained Random Forest remain
separate integration priorities. The local redesign does not redeploy AWS.
