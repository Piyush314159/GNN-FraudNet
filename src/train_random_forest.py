"""Train the retained Random Forest recipe without re-running rejected models.

Usage: python -m src.train_random_forest --output results/random_forest_retrain
Training periods 1–34, tuning 35–37, development evaluation 38–41.
Observed final-test periods 42–49 are excluded. This does not change the API.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parent.parent


def load_local_data(raw_dir):
    """Load 93 local attributes, aligned labels, and development periods only."""
    raw_dir = Path(raw_dir)
    raw = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None, usecols=range(95))
    if raw[0].duplicated().any():
        raise ValueError("Transaction IDs must be unique")
    times = raw[1].to_numpy(dtype=float)
    if not np.isfinite(times).all() or not np.equal(times, np.floor(times)).all() or (times < 1).any():
        raise ValueError("Periods must be positive finite integers")
    raw = raw.loc[times <= 41]
    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv").set_index("txId")
    if classes.index.duplicated().any():
        raise ValueError("Class transaction IDs must be unique")
    y = classes["class"].astype(str).map({"1": 1, "2": 0, "unknown": -1}).reindex(raw[0])
    if y.isna().any():
        raise ValueError("Missing or invalid development labels")
    x = raw.iloc[:, 2:95].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite")
    return x, raw[1].to_numpy(dtype=int), y.to_numpy(dtype=int)


def split_masks(times, labels):
    known = labels >= 0
    masks = (known & (times <= 34),
             known & (times > 34) & (times <= 37),
             known & (times > 37) & (times <= 41))
    if any(set(np.unique(labels[m])) != {0, 1} for m in masks):
        raise ValueError("Training, tuning and evaluation each require both classes")
    return masks


def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def metrics(y, p, threshold):
    pred = p >= threshold
    return dict(pr_auc=float(average_precision_score(y, p)),
                f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                brier=float(brier_score_loss(y, p)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                threshold=float(threshold))


def train_forest(x, times, labels, seed=42, n_estimators=500):
    train, tune, evaluate = split_masks(times, labels)
    scaler = QuantileTransformer(n_quantiles=min(1000, int(train.sum())),
                                 output_distribution="normal", subsample=None, random_state=42)
    scaler.fit(x[train])
    features = scaler.transform(x).astype(np.float32)
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2,
                                   class_weight="balanced", n_jobs=4, random_state=seed)
    model.fit(features[train], labels[train])
    tune_p = model.predict_proba(features[tune])[:, 1]
    threshold = choose_threshold(labels[tune], tune_p)
    eval_p = model.predict_proba(features[evaluate])[:, 1]
    report = dict(protocol="retained_random_forest_recipe", seed=seed,
                  n_estimators=n_estimators, production_changed=False,
                  preprocessing="Train-only quantile-to-normal; 93 local attributes",
                  split={name: dict(total=int(mask.sum()), illicit=int((labels[mask] == 1).sum()))
                         for name, mask in zip(("train", "tune", "evaluate"), (train, tune, evaluate))},
                  tuning=metrics(labels[tune], tune_p, threshold),
                  evaluation=metrics(labels[evaluate], eval_p, threshold),
                  limitations=["Previously observed development periods; not a fresh final test",
                               "Feature publication times have not been independently verified"])
    return model, scaler, threshold, eval_p, report


def save_run(output, model, scaler, threshold, probabilities, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "selected_model"
    artifact.mkdir()
    joblib.dump(model, artifact / "model.joblib")
    joblib.dump(scaler, artifact / "scaler.joblib")
    np.save(artifact / "evaluation_probabilities.npy", probabilities)
    (artifact / "metadata.json").write_text(json.dumps(dict(
        model="random_forest", threshold=threshold, seed=report["seed"],
        n_features=93, raw_feature_slice=[1, 94], train_end=34, tune_end=37,
        evaluated_through=41, production_ready=False), indent=2) + "\n")
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a fresh directory to preserve saved models")
    model, scaler, threshold, probabilities, report = train_forest(*load_local_data(args.raw_dir), seed=args.seed)
    report["packages"] = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "joblib")}
    report["data_sha256"] = {}
    for name in ("elliptic_txs_features.csv", "elliptic_txs_classes.csv"):
        digest = hashlib.sha256()
        with (args.raw_dir / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        report["data_sha256"][name] = digest.hexdigest()
    save_run(args.output, model, scaler, threshold, probabilities, report)
    print(json.dumps(report["evaluation"], indent=2))


if __name__ == "__main__":
    main()
