import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)


def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "brier_score": brier_score_loss(y_test, probs),
    }


def evaluate_models(models, X_test, y_test, threshold=0.5):
    rows = []

    for name, model in models.items():
        metrics = evaluate_classifier(model, X_test, y_test, threshold)
        metrics["model"] = name
        rows.append(metrics)

    cols = ["model", "roc_auc", "accuracy", "precision", "recall", "f1", "brier_score"]
    return pd.DataFrame(rows)[cols]


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]

        if i == 0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob > lower) & (y_prob <= upper)

        if mask.sum() > 0:
            avg_confidence = y_prob[mask].mean()
            avg_accuracy = y_true[mask].mean()
            bin_weight = mask.mean()
            ece += bin_weight * abs(avg_confidence - avg_accuracy)

    return ece


def build_prediction_analysis_df(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    analysis_df = X_test.copy()
    analysis_df["true"] = np.array(y_test)
    analysis_df["pred"] = preds
    analysis_df["prob_disease"] = probs
    analysis_df["confidence"] = np.maximum(probs, 1 - probs)
    analysis_df["correct"] = analysis_df["true"] == analysis_df["pred"]

    return analysis_df


def get_high_confidence_errors(analysis_df, min_confidence=0.80):
    return analysis_df[
        (analysis_df["correct"] == False)
        & (analysis_df["confidence"] >= min_confidence)
    ].sort_values("confidence", ascending=False)