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

TODO: Implement all functions below
"""

def train_logreg(X_train, y_train):
    # TODO: fit logistic regression with balanced class weights
    pass

def train_xgboost(X_train, y_train):
    # TODO: fit xgboost with scale_pos_weight for imbalance
    pass

def predict_proba(model, X):
    # TODO: return illicit class probabilities
    pass

def evaluate_baseline(model, X_test, y_test, model_name):
    # TODO: compute and return F1, PR-AUC, confusion matrix
    pass
