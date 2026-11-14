# Rolling model comparison: results

**Retained research candidate: Random Forest. Production API unchanged.**

Cleanup status: Random Forest's saved files and these comparison records remain.
Retired experiment code and superseded first-round weights have been removed.
Use the [current training workflow](../../../docs/random-forest.md) for the
retained recipe. Source fingerprints and software-test counts below describe
the original experiment, including files/tests retired afterward.

Five model configurations completed three historical windows and two seeds each. The official FT-Transformer completed three runs and was rejected using a numerical upper bound on the existing consistency criterion. No overall FT-Transformer average is claimed.

## Average evaluation scores

| Model | Mean AP | Mean fraud F1 | Outcome |
|---|---:|---:|---|
| XGBoost | 0.8862 | 0.8469 | Reference, superseded |
| Random Forest | 0.8902 | 0.8616 | **Kept** |
| Official MLP | 0.8692 | 0.8087 | Not retained |
| Official ResNet | 0.8751 | 0.8169 | Not retained |
| GraphSAGE + XGBoost | 0.8794 | 0.8429 | Not retained |
| Official FT-Transformer (compact configuration) | — | — | Rejected by consistency bound |

AP means average precision, consistent with the project PR-AUC convention. Scores are unweighted means across the six window/seed runs; they describe the configuration across separately trained models, not one model evaluated six times.

## Window-level comparison

| Evaluation periods | XGBoost AP | Random Forest AP | XGBoost F1 | Random Forest F1 |
|---|---:|---:|---:|---:|
| 24–27 | 0.7848 | 0.7720 | 0.7643 | 0.7546 |
| 31–34 | 0.9681 | 0.9909 | 0.9094 | 0.9447 |
| 38–41 | 0.9056 | 0.9078 | 0.8669 | 0.8854 |

Random Forest increases mean AP by 0.0040 and mean fraud F1 by 0.0147. It passes the predeclared screening gate: AP gain at least 0.002, no mean F1 regression, and AP non-regression in at least two windows. It performs worse in the first window, so the advantage is modest and period-dependent, not evidence of universal superiority.

## FT-Transformer early rejection

The researchers’ official implementation was used with two blocks, width 32 and eight attention heads. It is the FT-Transformer architecture, not the paper’s default-size configuration.
- Window 1, seed 42: AP 0.7452, fraud F1 0.7282.
- Window 1, seed 123: AP 0.7466, fraud F1 0.7094.
- Window 2, seed 42: AP 0.9000, fraud F1 0.7523.

Even assigning AP=1.0 to every remaining seed, the best attainable window AP values are 0.7459, 0.9500, and 1.0000. Only one window can still match Random Forest; two are required. The remaining runs cannot change the rejection, so the verified training process was stopped. No complete three-window FT-Transformer average was computed.

## Retained artifact and checks

- Only the Random Forest checkpoint from the final window, preselected seed 42, is retained from this round, together with its train-fitted preprocessing, tuning-selected threshold and metadata. Rejected candidate weights were not retained. Historical round-one metrics remain; superseded weights were removed during cleanup.
- Reloaded artifact predictions agree within floating-point rounding precision (maximum probability difference about 1e-16); classifications and reported metrics reproduce exactly.
- At experiment time, all 53 software tests passed, including chronological masks, removal of old test nodes/edges, official neural training/reloading, and the conservative pruning bound. Tests for retired experimental code were subsequently removed.
- Dataset hashes match round one, and source/configuration fingerprints are recorded in `comparison.json`. Artifact hashes are in `artifact_verification.json`.

## Limits

These are development comparisons on previously observed historical periods. Periods 42–49 were excluded and not rescored. There is no new independent final test. The two seeds and three overlapping historical windows do not establish statistical significance or live event-time performance. The new quantile preprocessing, tuning partitions and model settings differ from round one; do not compare the numerical averages directly with that round’s scores.

The API still uses GraphSAGE. Integrating the retained candidate requires its preprocessing, threshold, consistent label policy and matching tree-model explanations. See the [full protocol](../../../docs/rolling-comparison.md) and [official model implementation](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md).
