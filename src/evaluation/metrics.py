"""
Evaluation metrics for bridge contract prediction.

Computes: accuracy, precision, recall, F1 (macro),
          top-k accuracy, per-class report, confusion matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    label_encoder: LabelEncoder,
    top_k: list[int] = [3, 5],
    model_name: str = "",
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true        : true integer labels
        y_pred        : predicted integer labels
        y_proba       : predicted probabilities shape (n_samples, n_classes)
        label_encoder : fitted LabelEncoder for class names
        top_k         : list of k values for top-k accuracy
        model_name    : used for display

    Returns:
        dict of metric_name -> value
    """
    class_names = list(label_encoder.classes_)

    results: dict = {"model": model_name}

    # Core metrics
    results["accuracy"] = float(accuracy_score(y_true, y_pred))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    results["precision_macro"] = float(prec)
    results["recall_macro"] = float(rec)
    results["f1_macro"] = float(f1)

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    results["precision_weighted"] = float(prec_w)
    results["recall_weighted"] = float(rec_w)
    results["f1_weighted"] = float(f1_w)

    # Top-k accuracy
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        all_labels = list(range(n_classes))
        for k in top_k:
            if k < n_classes:
                results[f"top_{k}_accuracy"] = float(
                    top_k_accuracy_score(y_true, y_proba, k=k, labels=all_labels)
                )

    # Use all possible labels so reports are consistent across splits
    all_labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    results["per_class_report"] = report

    results["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=all_labels
    ).tolist()
    results["class_names"] = class_names

    return results


def print_summary(results: dict) -> None:
    """Print a concise summary of evaluation results."""
    model = results.get("model", "Model")
    print(f"\n{'='*55}")
    print(f"  {model}")
    print(f"{'='*55}")
    print(f"  Accuracy          : {results['accuracy']:.4f}")
    print(f"  Precision (macro) : {results['precision_macro']:.4f}")
    print(f"  Recall (macro)    : {results['recall_macro']:.4f}")
    print(f"  F1 (macro)        : {results['f1_macro']:.4f}")
    print(f"  F1 (weighted)     : {results['f1_weighted']:.4f}")
    for k in [3, 5]:
        key = f"top_{k}_accuracy"
        if key in results:
            print(f"  Top-{k} Accuracy    : {results[key]:.4f}")
    print(f"{'='*55}")


def save_results(results: dict, path: str | Path) -> None:
    """Save results dict to JSON (excludes heavy objects for readability)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saveable = {
        k: v for k, v in results.items()
        if k not in ("confusion_matrix", "per_class_report")
    }
    path.write_text(json.dumps(saveable, indent=2))


def compare_models(results_list: list[dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of result dicts."""
    metrics = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "f1_weighted", "top_3_accuracy", "top_5_accuracy",
    ]
    rows = []
    for res in results_list:
        row = {"model": res["model"]}
        for m in metrics:
            row[m] = res.get(m, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
