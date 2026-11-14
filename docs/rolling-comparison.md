# Second model comparison: rolling historical windows

This round revisits the previously discussed alternatives using the
[researchers' official MLP, ResNet and FT-Transformer implementation](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md).
The FT-Transformer uses its feature tokenizer, classification token, attention,
and feed-forward blocks, rather than the first experiment's mean-pooled generic
Transformer. Its CPU-sized configuration (two blocks, width 32, eight heads) is
explicitly recorded; it is not the paper's default configuration or a claim to
reproduce the paper's benchmark scores.

## Archived experiment

The comparison is complete. Its retired sweep runner, rejected model code and
optional neural-research dependency were removed during cleanup. The original
metrics, configurations, source fingerprints and rejection proof remain under
`results/model_selection/rolling_v2/`. Those fingerprints describe the code at
experiment time, including files that were subsequently removed.

The [current Random Forest workflow](random-forest.md) reproduces the retained
final-window recipe without re-running rejected candidates.

## Protocol declared before training

| Window | Training periods | Inner tuning periods | Evaluation periods |
|---|---|---|---|
| 1 | 1–20 | 21–23 | 24–27 |
| 2 | 1–27 | 28–30 | 31–34 |
| 3 | 1–34 | 35–37 | 38–41 |

- Seeds: 42 and 123. Identical deterministic XGBoost results across seeds are
  expected with its current full-sampling configuration.
- Periods 42–49 are physically excluded from all arrays and graph snapshots
  after loading. They are not rescored or used to choose models or thresholds.
- Features: the same 93 local attributes as the first experiment, excluding
  timestep and supplied aggregate neighborhood attributes.
- Every model receives the same train-fitted quantile-to-normal transformation.
  This reduces the influence of extreme values on neural training. Each window
  fits its own transformer on labelled training nodes, with a fixed preprocessing
  seed. Preprocessing is not fitted on tuning/evaluation nodes or unknown nodes.
- Neural early stopping and F1 threshold selection use only inner tuning labels.
  Evaluation predictions use the frozen threshold and selected weights.
- Evaluation labels never participate in gradient updates, threshold tuning, or
  early stopping. They are used to compare candidate architectures.
- Graph training sees the training-period snapshot. Hybrid training embeddings
  use that same snapshot. Tuning/evaluation embeddings are computed separately
  per period, excluding future graph endpoints.
- Candidate order: XGBoost reference, Random Forest (500 trees), official MLP,
  official tabular ResNet, GraphSAGE + XGBoost, official FT-Transformer.
- Neural budget: at most 30 epochs; stop after six non-improving tuning epochs.
  GraphSAGE uses its existing eight-epoch patience. All settings and timings are
  recorded. These are fixed configurations, not an exhaustive tuning search.
- Replace the incumbent only when mean evaluation average precision increases
  by at least 0.002, mean fraud F1 does not decrease, and average precision does
  not decrease in at least two of three windows (averaging seeds per window).
  This screening rule is not a statistical significance test.

## Safe early rejection

A resource-saving check was added while FT-Transformer was running. It does
not change the acceptance criterion. The check assigns AP=1.0 to every missing
seed result and calculates the best possible average for each window. If fewer
than two windows can still match the incumbent, the candidate cannot meet the
existing consistency requirement, regardless of its remaining results.

The saved FT-Transformer experiment entry includes `pruning_proof`, with the
completed runs, incumbent scores and numerical upper bounds. The helper script
was retired after the proof was recorded. No overall average was invented for
the uncompleted candidate; completing its remaining runs cannot reverse the
proven consistency failure.

## Retention and interpretation

`comparison.json` records every window/seed result, frozen thresholds, counts,
network configurations, package versions, dataset/source hashes and decisions.
Progress is saved after each run. Rejected candidate weights are not serialized.
Only the current winning candidate's last-window, seed-42 checkpoint is retained,
with its preprocessing and metadata. A new winner supersedes the previous new
checkpoint within this run. Seed 42 is fixed before evaluation, rather than
chosen for the highest score. The selected Random Forest files and comparison
records are preserved. Superseded first-round XGBoost files were removed during
cleanup; source fingerprints remain historical evidence, not live file links.

This is a development comparison on previously observed historical data. It is
not a new independent final test. Overlapping expanding training windows also
mean that six runs are not six independent datasets. The means are unweighted
across window/seed runs, not pooled across transactions; the reported spread
mixes period shifts and seed variation.

The first experiment used different preprocessing and tuning/evaluation blocks,
so its scores should not be compared directly with these averages. Both rounds
have period-level graph timing and precomputed local-feature limitations.
Neither establishes live event-time performance or publication-time availability.
The API remains unchanged; integration requires matching explanations,
preprocessing, label policy, and further robustness checks.
