"""Robustness analyses for the DDS/noDDS binary classification study.

Run from the repository root with .venv/Scripts/python.exe.
Outputs are confined to this experiment directory.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

RNG_SEED = 20261021
BOOTSTRAPS = 500
SHAP_N = 50
DATASETS = {"DDS": "processed", "noDDS": "processed_no_dds"}
ALGORITHMS = ("RandomForest", "XGBoost", "LightGBM")


def get_splits(variant: str):
    folder = ROOT / "data" / DATASETS[variant]
    features = json.loads((folder / "feature_columns.json").read_text(encoding="utf-8"))
    parts = {name: pd.read_csv(folder / f"{name}.csv") for name in ("train", "val", "test")}
    return parts, features


def load_baselines():
    result = {}
    for variant in DATASETS:
        for algorithm in ALGORITHMS:
            suffix = "_dds" if variant == "DDS" else ""
            path = ROOT / "outputs" / "models" / f"{algorithm.lower()}{suffix}.pkl"
            with path.open("rb") as handle:
                result[(variant, algorithm)] = pickle.load(handle)
    return result


def positive_probability(model, X):
    p = np.asarray(model.predict_proba(X))
    return p[:, 1] if p.ndim == 2 else p


def metrics(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(np.int8)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision_positive": float(precision_score(y, pred, zero_division=0)),
        "recall_positive": float(recall_score(y, pred, zero_division=0)),
        "f1_positive": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "tn": int(confusion_matrix(y, pred, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y, pred, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y, pred, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y, pred, labels=[0, 1])[1, 1]),
    }


def choose_f1_threshold(y, proba):
    grid = np.arange(0.05, 0.951, 0.005)
    scores = np.array([f1_score(y, proba >= t, average="macro", zero_division=0) for t in grid])
    best = np.flatnonzero(scores == scores.max())
    return float(grid[best[len(best) // 2]])


def make_clusters(groups):
    frame = pd.DataFrame({"group": groups})
    return [np.asarray(ix, dtype=np.int64) for ix in frame.groupby("group", sort=False).indices.values()]


def sample_cluster_indices(clusters, rng):
    selections = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in selections])


def bootstrap_ci(y, proba, threshold, groups, rng, n_boot=BOOTSTRAPS):
    samples = {key: [] for key in ("accuracy", "f1_macro", "average_precision")}
    # Deals, not rows, are independent resampling units. Labels may differ within a deal.
    clusters = make_clusters(groups)
    for _ in range(n_boot):
        ids = sample_cluster_indices(clusters, rng)
        row = metrics(y[ids], proba[ids], threshold)
        for key in samples:
            samples[key].append(row[key])
    return {key: [float(x) for x in np.percentile(values, [2.5, 97.5])] for key, values in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-inference", action="store_true", help="Reuse completed bootstrap/paired results and run SHAP plus source-held-out analysis")
    parser.add_argument("--statistics-only", action="store_true", help="Run only bootstrap and paired statistical analyses")
    parser.add_argument("--source-only", action="store_true", help="Run only the internal LIN-to-PBN source-held-out evaluation")
    args = parser.parse_args()
    print("Loading paired splits and baseline models...", flush=True)
    splits, features = {}, {}
    for variant in DATASETS:
        splits[variant], features[variant] = get_splits(variant)
    for part in ("train", "val", "test"):
        left = splits["DDS"][part]
        right = splits["noDDS"][part]
        identity = ["_source_file", "_room", "_board_number", "_deal_key", "label"]
        if not left[identity].equals(right[identity]):
            raise ValueError(f"DDS/noDDS {part} rows are not paired and identically ordered")
    models = load_baselines()
    y_val = splits["DDS"]["val"]["label"].to_numpy(dtype=np.int8)
    y_test = splits["DDS"]["test"]["label"].to_numpy(dtype=np.int8)
    groups_test = splits["DDS"]["test"]["_deal_key"].to_numpy()
    if not np.array_equal(y_val, splits["noDDS"]["val"]["label"].to_numpy(dtype=np.int8)):
        raise ValueError("Validation labels differ between DDS and noDDS")
    if not np.array_equal(y_test, splits["noDDS"]["test"]["label"].to_numpy(dtype=np.int8)):
        raise ValueError("Test labels differ between DDS and noDDS")

    if not args.skip_inference and not args.source_only:
        val_p, test_p, thresholds = {}, {}, {}
        threshold_rows, ci_rows = [], []
        rng = np.random.default_rng(RNG_SEED)
        for variant in DATASETS:
            X_val = splits[variant]["val"][features[variant]].to_numpy(dtype=np.float32)
            X_test = splits[variant]["test"][features[variant]].to_numpy(dtype=np.float32)
            for algorithm in ALGORITHMS:
                key = (variant, algorithm)
                model = models[key]
                val_p[key] = positive_probability(model, X_val)
                test_p[key] = positive_probability(model, X_test)
                threshold = choose_f1_threshold(y_val, val_p[key])
                thresholds[key] = threshold
                at_default = metrics(y_test, test_p[key], 0.5)
                at_tuned = metrics(y_test, test_p[key], threshold)
                threshold_rows.append({
                    "dataset": variant, "algorithm": algorithm,
                    "validation_selected_threshold": threshold,
                    **{f"default_{k}": v for k, v in at_default.items()},
                    **{f"validation_threshold_{k}": v for k, v in at_tuned.items()},
                    "validation_f1_macro_at_selected_threshold": metrics(y_val, val_p[key], threshold)["f1_macro"],
                })
                ci = bootstrap_ci(y_test, test_p[key], 0.5, groups_test, rng)
                row = {"dataset": variant, "algorithm": algorithm, **at_default}
                for metric_name, limits in ci.items():
                    row[f"{metric_name}_ci95_low"], row[f"{metric_name}_ci95_high"] = limits
                ci_rows.append(row)
                print(f"Threshold {variant}/{algorithm}: {threshold:.3f}; test AP={at_default['average_precision']:.4f}", flush=True)

        pd.DataFrame(threshold_rows).to_csv(OUT / "threshold_validation_test.csv", index=False)
        pd.DataFrame(ci_rows).to_csv(OUT / "baseline_test_bootstrap_ci.csv", index=False)

        paired_rows = []
        clusters = make_clusters(groups_test)
        preds = {key: (p >= 0.5) for key, p in test_p.items()}
        for algorithm in ALGORITHMS:
            p_dds, p_nodds = test_p[("DDS", algorithm)], test_p[("noDDS", algorithm)]
            pred_dds, pred_nodds = preds[("DDS", algorithm)], preds[("noDDS", algorithm)]
            dds_correct, nodds_correct = pred_dds == y_test, pred_nodds == y_test
            per_group = pd.DataFrame({"group": groups_test, "dds": dds_correct.astype(int), "nodds": nodds_correct.astype(int)})
            delta_by_group = per_group.groupby("group", sort=False).apply(lambda g: g["dds"].mean() - g["nodds"].mean(), include_groups=False)
            b_groups = int((delta_by_group > 0).sum())
            c_groups = int((delta_by_group < 0).sum())
            pvalue = float(binomtest(min(b_groups, c_groups), b_groups + c_groups, 0.5).pvalue) if b_groups + c_groups else 1.0
            rng_pair = np.random.default_rng(RNG_SEED + ALGORITHMS.index(algorithm))
            diffs = {m: [] for m in ("accuracy", "f1_macro", "average_precision")}
            for _ in range(BOOTSTRAPS):
                ids = sample_cluster_indices(clusters, rng_pair)
                md = metrics(y_test[ids], p_dds[ids], 0.5)
                mn = metrics(y_test[ids], p_nodds[ids], 0.5)
                for m in diffs:
                    diffs[m].append(md[m] - mn[m])
            row = {
                "algorithm": algorithm,
                "accuracy_delta_dds_minus_nodds": accuracy_score(y_test, pred_dds) - accuracy_score(y_test, pred_nodds),
                "f1_macro_delta_dds_minus_nodds": f1_score(y_test, pred_dds, average="macro") - f1_score(y_test, pred_nodds, average="macro"),
                "ap_delta_dds_minus_nodds": average_precision_score(y_test, p_dds) - average_precision_score(y_test, p_nodds),
                "paired_bootstrap_accuracy_delta_ci95_low": float(np.percentile(diffs["accuracy"], 2.5)),
                "paired_bootstrap_accuracy_delta_ci95_high": float(np.percentile(diffs["accuracy"], 97.5)),
                "paired_bootstrap_f1_macro_delta_ci95_low": float(np.percentile(diffs["f1_macro"], 2.5)),
                "paired_bootstrap_f1_macro_delta_ci95_high": float(np.percentile(diffs["f1_macro"], 97.5)),
                "paired_bootstrap_ap_delta_ci95_low": float(np.percentile(diffs["average_precision"], 2.5)),
                "paired_bootstrap_ap_delta_ci95_high": float(np.percentile(diffs["average_precision"], 97.5)),
                "deal_groups_dds_higher_accuracy": b_groups,
                "deal_groups_nodds_higher_accuracy": c_groups,
                "deal_groups_tied_accuracy": int((delta_by_group == 0).sum()),
                "exact_sign_test_pvalue_deal_level_accuracy": pvalue,
            }
            paired_rows.append(row)
        paired_df = pd.DataFrame(paired_rows)
        order = np.argsort(paired_df["exact_sign_test_pvalue_deal_level_accuracy"].to_numpy())
        adjusted = np.empty(len(order), dtype=float)
        running_max = 0.0
        raw_p = paired_df["exact_sign_test_pvalue_deal_level_accuracy"].to_numpy()
        for rank, idx in enumerate(order):
            running_max = max(running_max, (len(order) - rank) * raw_p[idx])
            adjusted[idx] = min(1.0, running_max)
        paired_df["holm_adjusted_sign_test_pvalue"] = adjusted
        paired_df.to_csv(OUT / "paired_dds_vs_nodds_inference.csv", index=False)
        print("Deal-cluster bootstrap and exact deal-level sign tests saved.", flush=True)

    if not args.statistics_only and not args.source_only:
        print("Computing common-method TreeSHAP on the same validation sample...", flush=True)
        import shap
        sample_ids = np.random.default_rng(RNG_SEED).choice(len(y_val), size=min(SHAP_N, len(y_val)), replace=False)
        shap_rows = []
        for variant in DATASETS:
            X = splits[variant]["val"][features[variant]].to_numpy(dtype=np.float32)[sample_ids]
            for algorithm in ALGORITHMS:
                core = getattr(models[(variant, algorithm)], "clf", models[(variant, algorithm)])
                explainer = shap.TreeExplainer(core)
                raw = explainer.shap_values(X)
                if isinstance(raw, list):
                    values = np.asarray(raw[1] if len(raw) > 1 else raw[0])
                else:
                    values = np.asarray(raw)
                    if values.ndim == 3:
                        values = values[:, :, 1] if values.shape[0] == len(X) else values[1]
                if values.ndim != 2:
                    raise ValueError(f"Unexpected SHAP dimensions {values.shape} for {variant}/{algorithm}")
                importance = np.abs(values).mean(axis=0)
                dds_mask = np.array([f.startswith(("ns_dd_", "ew_dd_")) for f in features[variant]])
                dds_share = float(importance[dds_mask].sum() / importance.sum()) if dds_mask.any() else 0.0
                order = np.argsort(importance)[::-1][:10]
                shap_rows.append({
                    "dataset": variant, "algorithm": algorithm, "n_validation_rows": len(sample_ids),
                "method": "TreeExplainer default (exact Tree SHAP; LightGBM native implementation)",
                    "dds_feature_mean_abs_shap_share": dds_share,
                    "top_10_features": ", ".join(features[variant][i] for i in order),
                })
                print(f"SHAP {variant}/{algorithm}: DDS share {dds_share:.2%}", flush=True)
        pd.DataFrame(shap_rows).to_csv(OUT / "matched_exact_tree_shap.csv", index=False)

    if not args.statistics_only:
        print("Running source-held-out evaluation within the merged dataset: train on LIN, hold out PBN...", flush=True)
        source_rows = []
        cfg = json.loads("{}")
        import yaml
        cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))["models"]
        for variant in DATASETS:
            folder = ROOT / "data" / DATASETS[variant]
            full = pd.read_csv(folder / "full.csv")
            is_pbn = full["_source_file"].astype(str).str.lower().str.endswith(".pbn")
            train_all, heldout = full.loc[~is_pbn].copy(), full.loc[is_pbn].copy()
            feat = features[variant]
            y_lin = train_all["label"].to_numpy(dtype=np.int8)
            if train_all.empty or heldout.empty:
                raise ValueError(f"Missing source domain for {variant}")
            split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
            inner_tr, inner_va = next(split.split(np.zeros(len(y_lin)), y_lin, groups=train_all["_deal_key"].to_numpy()))
            for algorithm in ALGORITHMS:
                block = cfg[{"RandomForest":"random_forest", "XGBoost":"xgboost", "LightGBM":"lightgbm"}[algorithm]].copy()
                block.pop("n_jobs", None); block.pop("random_state", None); block.pop("class_weight", None)
                if algorithm == "RandomForest":
                    model = RandomForestClassifier(**block, class_weight="balanced", random_state=42, n_jobs=-1)
                elif algorithm == "XGBoost":
                    model = XGBClassifier(**block, random_state=42, n_jobs=-1, verbosity=0, eval_metric="logloss")
                else:
                    model = LGBMClassifier(**block, class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1)
                X_lin = train_all[feat].to_numpy(dtype=np.float32)
                model.fit(X_lin[inner_tr], y_lin[inner_tr])
                p_inner_va = positive_probability(model, X_lin[inner_va])
                threshold = choose_f1_threshold(y_lin[inner_va], p_inner_va)
                # Refit on all LIN records only; PBN labels are never used for fitting or threshold choice.
                model.fit(X_lin, y_lin)
                X_pbn = heldout[feat].to_numpy(dtype=np.float32)
                p_pbn = positive_probability(model, X_pbn)
                base = metrics(heldout["label"].to_numpy(dtype=np.int8), p_pbn, 0.5)
                tuned = metrics(heldout["label"].to_numpy(dtype=np.int8), p_pbn, threshold)
                source_rows.append({
                    "dataset": variant, "algorithm": algorithm,
                    "train_source": "LIN only", "heldout_source": "PBN only",
                    "n_train_lin": len(train_all), "n_heldout_pbn": len(heldout),
                    "lin_positive_rate": float(y_lin.mean()),
                    "pbn_positive_rate": float(heldout["label"].mean()),
                    "threshold_selected_on_lin_validation": threshold,
                    **{f"default_{k}": v for k, v in base.items()},
                    **{f"lin_threshold_{k}": v for k, v in tuned.items()},
                })
                print(f"Source holdout {variant}/{algorithm}: AP={base['average_precision']:.4f}, tuned F1+={tuned['f1_positive']:.4f}", flush=True)
        pd.DataFrame(source_rows).to_csv(OUT / "source_heldout_lin_to_pbn.csv", index=False)

    metadata = {
        "created": "2026-10-21",
        "test_bootstrap_replicates": BOOTSTRAPS,
        "shap_sample_size": SHAP_N,
        "shap_method": "TreeExplainer default exact Tree SHAP path-dependent method across the six models; LightGBM uses its native Tree SHAP implementation",
        "threshold_selection": "maximize macro-F1 over validation probabilities, threshold grid 0.05 to 0.95 in steps of 0.005",
        "source_heldout_design": "internal source-held-out assessment on the already merged LIN+PBN dataset: train on LIN rows only; choose threshold on grouped 20% LIN validation using deal-aware StratifiedGroupKFold; refit on all LIN; evaluate on the PBN subset",
        "source_heldout_scope": "source-separated internal validation within the merged LIN+PBN dataset; not independent external validation",
        "paired_test_design": "deal-cluster bootstrap DDS minus noDDS and exact sign test on deal-level accuracy differences; test set used only for final estimation",
        "bootstrap_unit": "_deal_key groups sampled with replacement; within-deal rows remain together, class labels can differ within a deal",
        "positive_class_rate_test": float(y_test.mean()),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Completed. Results: {OUT}", flush=True)


if __name__ == "__main__":
    main()



