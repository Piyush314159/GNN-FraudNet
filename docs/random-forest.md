# Retained Random Forest workflow

The selected research model is stored in
`results/model_selection/rolling_v2/selected_model/`, with its preprocessing,
decision threshold, metadata and evaluation probabilities. The active API still
uses GraphSAGE and its matching explanations; its checkpoint remains required.

To train the retained recipe into a fresh directory:

```bash
.venv/bin/python -m src.train_random_forest --output results/random_forest_retrain
```

The script refuses to overwrite an existing output. It uses:

- 93 local attributes from raw columns 2–94, equivalent to `features[1:94]`
  in a 166-feature API request; excludes timestep and supplied aggregates.
- Periods 1–34 for training, 35–37 for tuning, and 38–41 for evaluation.
  Previously observed final-test periods 42–49 are excluded.
- A quantile-to-normal transformation fitted only on labelled training nodes.
- Random Forest with 500 trees, minimum leaf size 2, balanced class weights,
  four workers and seed 42. The preprocessing seed stays fixed at 42.
- A decision threshold selected by fraud F1 on the tuning block, then frozen
  for evaluation. Evaluation labels never affect model fitting or the threshold.

Outputs include `training.json` and a `selected_model/` directory. The report
records metrics, package versions and input-data hashes. These are development
results on previously observed periods, not a new independent final test.

The retired comparison scripts, standalone GAT/baseline implementations,
experimental neural networks and their dedicated tests/dependencies were removed.
Historical metrics, notebooks, figures comparing models and authored reports
remain as research records. The notebooks contain their own historical model
implementations and are not the current training entrypoint.

See [the comparison results](../results/model_selection/rolling_v2/summary.md),
[improvement priorities](improvement-roadmap.md), and
[cleanup record](../results/model_selection/cleanup.json).
