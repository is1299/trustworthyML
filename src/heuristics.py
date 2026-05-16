import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def clinician_heuristic(row):
    score = 0

    if row["age"] >= 55:
        score += 1
    if row["trestbps"] >= 140:
        score += 1
    if row["chol"] >= 240:
        score += 1
    if row["thalach"] < 140:
        score += 1
    if row["exang"] == 1:
        score += 1
    if row["oldpeak"] >= 2:
        score += 1
    if row["cp"] == 4:
        score += 1

    return 1 if score >= 3 else 0


def predict_heuristic(X):
    return X.apply(clinician_heuristic, axis=1)


def evaluate_heuristic(X_test, y_test):
    preds = predict_heuristic(X_test)

    return {
        "model": "Human-style heuristic",
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }


def add_heuristic_disagreement(analysis_df, X_test):
    heuristic_preds = predict_heuristic(X_test)

    df = analysis_df.copy()
    df["heuristic_pred"] = heuristic_preds.values
    df["ml_vs_heuristic_disagree"] = df["pred"] != df["heuristic_pred"]

    return df.sort_values("confidence", ascending=False)