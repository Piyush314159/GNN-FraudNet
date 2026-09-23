"""
baseline.py
===========
PURPOSE:
    Non-graph baseline models. These deliberately ignore graph structure
    and treat each node independently. Used to prove GNNs add value.

MODELS:
    1. Logistic Regression (sklearn)
    2. XGBoost (xgboost)

INPUTS:
    - X_train : node features for training nodes (numpy array)
    - y_train : labels for training nodes
    - X_test  : node features for test nodes
    - y_test  : labels for test nodes

HOW IT WORKS:
    train_logreg(X_train, y_train)
        Fit sklearn LogisticRegression with class_weight='balanced'
        Return: fitted model

    train_xgboost(X_train, y_train)
        Compute scale_pos_weight = count(licit) / count(illicit) for imbalance
        Fit XGBClassifier with that weight
        Return: fitted model

    predict_proba(model, X)
        Return probability of illicit class (column 1 of predict_proba output)

    evaluate_baseline(model, X_test, y_test, model_name)
        Compute F1 (illicit class), PR-AUC, confusion matrix
        Print results
        Return: dict of metrics

OUTPUT:
    Fitted model objects
    Metrics dict for comparison in evaluate.py
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from xgboost import XGBClassifier


def train_logreg(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    n_licit = (y_train == 0).sum()
    n_illicit = (y_train == 1).sum()
    scale_pos_weight = n_licit / max(n_illicit, 1)

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        n_estimators=200,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def evaluate_baseline(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = predict_proba(model, X_test)

    f1 = f1_score(y_test, y_pred, pos_label=1)
    pr_auc = average_precision_score(y_test, y_prob, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{model_name} — F1 (illicit): {f1:.4f}, PR-AUC: {pr_auc:.4f}")

    return {
        "model_name": model_name,
        "f1": f1,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
