"""Build, train, and evaluate the current binary par-contract pipeline.

Uses the LIN + PBN dataset and DDS v4 cache. Splits are grouped by the
canonical 52-card deal so duplicate deals across sources cannot cross splits.
Run from the repository root with the project virtual environment.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import PBNParser  # noqa: E402
from src.preprocessing import build_dataset, load_splits  # noqa: E402
from src.models import RFModel, XGBModel, LGBMModel  # noqa: E402
from src.evaluation import evaluate, print_summary, save_results, compare_models  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
PROCESSED_DIR = ROOT / "data" / "processed"
DDS_CACHE = ROOT / "data" / "combined_dds_cache_v4.csv"
MODELS_DIR = ROOT / "outputs" / "models"
RESULTS_DIR = ROOT / "outputs" / "results"


def main() -> None:
    if not DDS_CACHE.exists():
        raise FileNotFoundError(
            f"Required label cache is missing: {DDS_CACHE}. "
            "Build it with scripts/build_dds_cache.py first."
        )

    pbn_boards = PBNParser().parse_directory(ROOT / "data" / "raw_pbn")
    print(f"Building binary dataset from LIN + {len(pbn_boards):,} PBN boards...")
    build_dataset(
        raw_dir=ROOT / "data" / "raw",
        output_dir=PROCESSED_DIR,
        target_col="matches_par_contract",
        train_ratio=CONFIG["data"]["train_ratio"],
        val_ratio=CONFIG["data"]["val_ratio"],
        test_ratio=CONFIG["data"]["test_ratio"],
        random_seed=CONFIG["data"]["random_seed"],
        include_dds=True,
        dds_cache_path=DDS_CACHE,
        extra_boards=pbn_boards,
    )

    df_train, df_val, df_test, feature_cols, label_encoder = load_splits(PROCESSED_DIR)
    X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    y_train = df_train["label"].to_numpy()
    X_val = df_val[feature_cols].to_numpy(dtype=np.float32)
    y_val = df_val["label"].to_numpy()
    X_test = df_test[feature_cols].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy()

    model_configs = CONFIG["models"]
    models = [
        RFModel(**model_configs["random_forest"]),
        XGBModel(**model_configs["xgboost"]),
        LGBMModel(**model_configs["lightgbm"]),
    ]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    val_results = []
    for model in models:
        print(f"\n[TRAIN] {model.name} ...")
        started = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - started
        model.save(MODELS_DIR / f"{model.name.lower()}.pkl")
        result = evaluate(y_val, model.predict(X_val), model.predict_proba(X_val), label_encoder, model.name)
        result["train_time_sec"] = round(elapsed, 2)
        val_results.append(result)
        print_summary(result)

    val_table = compare_models(val_results)
    val_table.to_csv(RESULTS_DIR / "val_comparison.csv")
    (RESULTS_DIR / "val_summary.json").write_text(
        json.dumps(val_results, indent=2), encoding="utf-8"
    )

    print("\n=== Final test evaluation (once, after training choices are fixed) ===")
    test_results = []
    for model in models:
        result = evaluate(
            y_test, model.predict(X_test), model.predict_proba(X_test),
            label_encoder, model.name,
        )
        test_results.append(result)
        save_results(result, RESULTS_DIR / f"{model.name.lower()}_test.json")
        print_summary(result)

    compare_models(test_results).to_csv(RESULTS_DIR / "test_comparison.csv")
    (RESULTS_DIR / "test_summary.json").write_text(
        json.dumps(test_results, indent=2), encoding="utf-8"
    )
    print(f"\nPipeline complete. Results saved under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
