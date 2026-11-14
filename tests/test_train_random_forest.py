import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.train_random_forest import (choose_threshold, load_local_data, metrics,
                                     save_run, split_masks, train_forest)


def tiny_data():
    rng = np.random.default_rng(7)
    times = np.array([1] * 8 + [35] * 4 + [38] * 4 + [20, 42])
    y = np.array([0, 1] * 8 + [-1, 1])
    x = rng.normal(size=(len(y), 93)).astype(np.float32)
    x[-2:] = 10000
    return x, times, y


def test_headerless_loader_aligns_labels_and_removes_observed_test_periods(tmp_path):
    raw = np.zeros((4, 167))
    raw[:, 0] = [10, 20, 30, 40]
    raw[:, 1] = [1, 35, 38, 42]
    raw[:, 2:95] = np.arange(93)
    raw[:, 95:] = 999
    pd.DataFrame(raw).to_csv(tmp_path / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame(dict(txId=[40, 30, 10, 20], **{"class": ["1", "unknown", "2", "1"]})).to_csv(
        tmp_path / "elliptic_txs_classes.csv", index=False)
    x, times, labels = load_local_data(tmp_path)
    assert x.shape == (3, 93)
    assert times.tolist() == [1, 35, 38]
    assert labels.tolist() == [0, 1, -1]
    assert x[0].tolist() == list(range(93))


def test_splits_are_disjoint_and_exclude_unknown_and_final_test_labels():
    _, times, y = tiny_data()
    train, tune, evaluate = split_masks(times, y)
    assert (train.sum(), tune.sum(), evaluate.sum()) == (8, 4, 4)
    assert not (train & tune | tune & evaluate | train & evaluate).any()
    assert not (train | tune | evaluate)[-2:].any()


def test_threshold_is_selected_from_tuning_predictions():
    y, p = np.array([0, 0, 1, 1]), np.array([.1, .2, .3, .4])
    threshold = choose_threshold(y, p)
    assert threshold == .3
    assert metrics(y, p, threshold)["f1"] == 1


def test_evaluation_labels_cannot_change_training_scaler_or_threshold():
    x, times, y = tiny_data()
    first = train_forest(x, times, y, n_estimators=8)
    changed = y.copy()
    changed[times >= 38] = 1 - changed[times >= 38]
    second = train_forest(x, times, changed, n_estimators=8)
    assert np.array_equal(first[1].quantiles_, second[1].quantiles_)
    assert first[2] == second[2]
    assert np.allclose(first[3], second[3])
    assert first[1].quantiles_.max() < 10000


def test_saved_model_and_preprocessor_reproduce_frozen_predictions(tmp_path):
    x, times, y = tiny_data()
    fitted = train_forest(x, times, y, n_estimators=8)
    output = tmp_path / "new_run"
    save_run(output, *fitted)
    artifact = output / "selected_model"
    model = joblib.load(artifact / "model.joblib")
    scaler = joblib.load(artifact / "scaler.joblib")
    meta = json.loads((artifact / "metadata.json").read_text())
    mask = split_masks(times, y)[2]
    p = model.predict_proba(scaler.transform(x[mask]).astype(np.float32))[:, 1]
    assert np.allclose(p, fitted[3])
    assert meta["threshold"] == fitted[2]
    with pytest.raises(FileExistsError):
        save_run(output, *fitted)


def test_each_split_requires_both_classes():
    x, times, y = tiny_data()
    y[times == 35] = 0
    with pytest.raises(ValueError, match="both classes"):
        train_forest(x, times, y, n_estimators=8)
