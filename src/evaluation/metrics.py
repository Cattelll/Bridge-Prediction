"""
Evaluation metrics for the binary "matches_par_contract" contract prediction task.

Computes: accuracy, precision/recall/F1 (macro, weighted, and for the
positive class), ROC-AUC, PR-AUC (average precision), per-class report,
confusion matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    label_encoder: LabelEncoder,
    model_name: str = "",
) -> dict:
    """
    Compute all evaluation metrics for the binary matches_par_contract target.

    Args:
        y_true        : true integer labels (0/1)
        y_pred        : predicted integer labels (0/1)
        y_proba       : predicted probabilities shape (n_samples, 2)
        label_encoder : fitted LabelEncoder for class names (classes_ == [0, 1])
        model_name    : used for display

    Returns:
        dict of metric_name -> value
    """
    class_names = [str(c) for c in label_encoder.classes_]

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

    # Positive-class ("optimal", label 1) metrics
    prec_pos, rec_pos, f1_pos, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    results["precision_positive"] = float(prec_pos)
    results["recall_positive"] = float(rec_pos)
    results["f1_positive"] = float(f1_pos)

    # Probability-based metrics
    if y_proba is not None:
        y_score = y_proba[:, 1]
        results["roc_auc"] = float(roc_auc_score(y_true, y_score))
        results["average_precision"] = float(average_precision_score(y_true, y_score))

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
    print(f"  F1 (positive)     : {results['f1_positive']:.4f}")
    if "roc_auc" in results:
        print(f"  ROC-AUC           : {results['roc_auc']:.4f}")
    if "average_precision" in results:
        print(f"  PR-AUC (avg prec) : {results['average_precision']:.4f}")
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
        "f1_weighted", "f1_positive", "roc_auc", "average_precision",
    ]
    rows = []
    for res in results_list:
        row = {"model": res["model"]}
        for m in metrics:
            row[m] = res.get(m, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
