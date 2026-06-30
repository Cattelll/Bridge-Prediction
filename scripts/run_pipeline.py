"""
Full training and evaluation pipeline.

Stages:
  1. Load processed dataset (or rebuild if missing)
  2. Train RF, XGBoost, LightGBM
  3. Evaluate on validation set
  4. Final evaluation on test set
  5. Save models + results

Run: py -3.12 scripts/run_pipeline.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import build_dataset, load_splits
from src.models import RFModel, XGBModel, LGBMModel
from src.evaluation import evaluate, print_summary, save_results, compare_models

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("outputs/models")
RESULTS_DIR = Path("outputs/results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    if not (PROCESSED_DIR / "train.csv").exists():
        print("Processed dataset not found — building...")
        build_dataset(raw_dir="data/raw", output_dir=PROCESSED_DIR)

    print("Loading dataset...")
    df_train, df_val, df_test, feature_cols, le = load_splits(PROCESSED_DIR)

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["label"].values

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["label"].values

    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test["label"].values

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Features: {len(feature_cols)}  Classes: {len(le.classes_)}")

    # ------------------------------------------------------------------
    # 2. Define models
    # ------------------------------------------------------------------
    models = [
        RFModel(),
        XGBModel(),
        LGBMModel(),
    ]

    # ------------------------------------------------------------------
    # 3. Train & evaluate on validation
    # ------------------------------------------------------------------
    val_results = []
    for model in models:
        print(f"\n[TRAIN] {model.name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"        Training time: {elapsed:.1f}s")

        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)

        res = evaluate(y_val, y_pred_val, y_proba_val, le, model_name=model.name)
        res["train_time_sec"] = round(elapsed, 2)
        val_results.append(res)
        print_summary(res)

        # Save model
        model.save(MODELS_DIR / f"{model.name.lower()}.pkl")

    # ------------------------------------------------------------------
    # 4. Final test evaluation (best model by val F1)
    # ------------------------------------------------------------------
    print("\n\n=== FINAL TEST EVALUATION ===")
    test_results = []
    for model, vres in zip(models, val_results):
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)
        res = evaluate(y_test, y_pred_test, y_proba_test, le, model_name=model.name)
        test_results.append(res)
        print_summary(res)
        save_results(res, RESULTS_DIR / f"{model.name.lower()}_test.json")

    # ------------------------------------------------------------------
    # 5. Comparison table
    # ------------------------------------------------------------------
    print("\n\n=== VALIDATION COMPARISON ===")
    val_df = compare_models(val_results)
    print(val_df.to_string(float_format=lambda x: f"{x:.4f}"))
    val_df.to_csv(RESULTS_DIR / "val_comparison.csv")

    print("\n=== TEST COMPARISON ===")
    test_df = compare_models(test_results)
    print(test_df.to_string(float_format=lambda x: f"{x:.4f}"))
    test_df.to_csv(RESULTS_DIR / "test_comparison.csv")

    # ------------------------------------------------------------------
    # 6. Feature importance (top 20 per model)
    # ------------------------------------------------------------------
    print("\n=== FEATURE IMPORTANCE (top 20) ===")
    for model in models:
        importances = model.feature_importances()
        if importances is None:
            continue
        top_idx = np.argsort(importances)[::-1][:20]
        print(f"\n  {model.name}:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feature_cols[idx]:35s} {importances[idx]:.4f}")

    print("\nPipeline complete. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
