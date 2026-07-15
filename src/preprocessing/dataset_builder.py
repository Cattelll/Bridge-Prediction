"""
Build a clean tabular dataset from parsed LIN boards.

Steps:
  1. Parse all LIN files → BoardRecord list
  2. Extract features per board
  3. Drop duplicates and validate
  4. Encode categorical target
  5. Split into train / validation / test
  6. Save CSV files and label encoder
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.parser import LINParser
from src.features import extract_features, CONTRACT_LABELS


# Columns used as ML features (all numeric/binary — no encoding needed)
_META_COLS = {
    "_board_number", "_source_file", "_room", "_declarer",
    "_result", "_tricks_made",
}
_TARGET_COLS = {"target", "target_base", "target_category"}


def _get_dds_dataframe(boards, dds_cache_path: str | Path | None = None) -> pd.DataFrame:
    """Load cached DDS features if available, else compute them (slow, ~1hr/500 files).

    Both paths yield the same raw schema (`dd_par_denom` as a single 0-4 int);
    `expand_par_denom()` one-hot-encodes it uniformly regardless of source.
    """
    from src.features.dds import expand_par_denom

    if dds_cache_path is not None and Path(dds_cache_path).exists():
        print(f"      Loading cached DDS features: {dds_cache_path}")
        df_dds = pd.read_csv(dds_cache_path)
    else:
        print("      No DDS cache found — computing from scratch (this is slow)...")
        from src.features.dds import compute_dds_features

        rows = []
        for b in boards:
            row = compute_dds_features(b)
            if row is None:
                continue
            row["_source_file"] = b.source_file
            row["_room"] = b.room
            row["_board_number"] = b.board_number
            rows.append(row)
        df_dds = pd.DataFrame(rows)

    return expand_par_denom(df_dds)


def build_dataset(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    target_col: str = "target_base",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    remove_pass: bool = False,
    include_dds: bool = False,
    dds_cache_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Parse LIN files, extract features, clean, encode, and split.

    Args:
        raw_dir        : directory containing .lin files
        output_dir     : where to save processed CSVs
        target_col     : which target to use ('target_base', 'target', 'target_category')
        train_ratio    : fraction for training
        val_ratio      : fraction for validation
        test_ratio     : fraction for test
        random_seed    : reproducibility seed
        remove_pass    : if True, drop passed-out boards from the dataset
        include_dds    : if True, add Double-Dummy Solver features (optional,
                         see CLAUDE.md scope — requires `endplay`)
        dds_cache_path : path to a precomputed DDS CSV (keyed by
                         _source_file/_room/_board_number) to avoid recomputing;
                         if None or missing, DDS is computed from scratch
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Parse & extract
    # ------------------------------------------------------------------
    print("[1/5] Parsing LIN files...")
    parser = LINParser()
    boards = parser.parse_directory(raw_dir)
    print(f"      Boards loaded: {len(boards)}")

    print("[2/5] Extracting features...")
    rows = []
    for b in boards:
        row = extract_features(b)
        if row is not None:
            rows.append(row)
    print(f"      Feature rows : {len(rows)}")

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 2. Cleaning
    # ------------------------------------------------------------------
    print("[3/5] Cleaning...")

    # Optionally remove passed-out boards
    if remove_pass:
        before = len(df)
        df = df[df[target_col] != "PASS"].copy()
        print(f"      Removed {before - len(df)} passed-out boards.")

    # Drop exact duplicates on feature columns only
    feature_cols = [c for c in df.columns if c not in _META_COLS and c not in _TARGET_COLS]
    before = len(df)
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    print(f"      Removed {before - len(df)} duplicate rows.")
    print(f"      Final rows   : {len(df)}")

    # Verify no missing values in feature columns
    missing = df[feature_cols].isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"[WARN] Missing values found:\n{missing}")
    else:
        print("      No missing values in features.")

    # Remove classes with <2 samples so all classes can appear in training after split
    # (also ensures XGBoost gets contiguous 0..n-1 labels)
    class_counts_pre = df[target_col].value_counts()
    rare = class_counts_pre[class_counts_pre < 2].index.tolist()
    if rare:
        print(f"[WARN] Removing {len(rare)} class(es) with <2 samples: {rare}")
        df = df[~df[target_col].isin(rare)].copy().reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Label encoding
    # ------------------------------------------------------------------
    print("[4/5] Encoding labels...")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[target_col])
    label_map = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    print(f"      Classes      : {len(le.classes_)}")
    print(f"      Classes list : {list(le.classes_)}")

    # Save encoder
    encoder_path = output_dir / "label_encoder.pkl"
    with open(encoder_path, "wb") as fp:
        pickle.dump(le, fp)
    json_path = output_dir / "label_map.json"
    json_path.write_text(json.dumps(label_map, indent=2))
    print(f"      Encoder saved: {encoder_path}")

    # ------------------------------------------------------------------
    # 3b. Optional: Double-Dummy Solver features
    # ------------------------------------------------------------------
    if include_dds:
        print("[DDS] Menambahkan fitur Double-Dummy Solver...")
        df_dds = _get_dds_dataframe(boards, dds_cache_path)
        dds_feature_cols = [
            c for c in df_dds.columns if c not in ("_source_file", "_room", "_board_number")
        ]
        before = len(df)
        df = df.merge(df_dds, on=["_source_file", "_room", "_board_number"], how="left")
        assert len(df) == before, "DDS join changed row count — identity key isn't unique"

        n_missing = df[dds_feature_cols].isnull().any(axis=1).sum()
        if n_missing:
            print(f"[WARN] {n_missing} row(s) missing DDS values — dropping them")
            df = df.dropna(subset=dds_feature_cols).reset_index(drop=True)

        feature_cols = feature_cols + dds_feature_cols
        print(f"      DDS fitur ditambahkan: {len(dds_feature_cols)} "
              f"(total fitur: {len(feature_cols)})")

    # ------------------------------------------------------------------
    # 4. Train / val / test split — GROUP-AWARE by physical deal
    # ------------------------------------------------------------------
    # BBO vugraph records each board twice ("open room" / "closed room"):
    # two different pairs bid the SAME dealt hands. That means ~all
    # hand-based features (HCP, shape, fit, stoppers, LTC, ...) are
    # identical between the two rows of a pair — only the auction/
    # contract differs. A plain random split let ~46% of these pairs
    # fall into different partitions, so ~60% of val/test rows had an
    # identical-hand "twin" sitting in train (verified empirically:
    # val/test accuracy on those rows was 92-95% when the twin bid the
    # SAME contract vs 23-28% when it bid a DIFFERENT one — the model
    # was partly memorizing deals, not generalizing). Grouping by
    # (_source_file, _board_number) keeps every pair on the same side
    # of the split, eliminating that leakage.
    print("[5/5] Splitting (group-aware by source_file + board_number)...")

    groups = df["_source_file"].astype(str) + "||" + df["_board_number"].astype(str)

    N_FOLDS = 20  # 14 train / 3 val / 3 test folds == 70% / 15% / 15%
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_seed)

    fold_of_row = np.empty(len(df), dtype=int)
    for fold_idx, (_, fold_idx_rows) in enumerate(sgkf.split(df, df["label"], groups)):
        fold_of_row[fold_idx_rows] = fold_idx

    n_train_folds = round(N_FOLDS * train_ratio)
    n_val_folds = round(N_FOLDS * val_ratio)
    train_folds = set(range(0, n_train_folds))
    val_folds = set(range(n_train_folds, n_train_folds + n_val_folds))
    test_folds = set(range(n_train_folds + n_val_folds, N_FOLDS))

    df_train = df[np.isin(fold_of_row, list(train_folds))].reset_index(drop=True)
    df_val = df[np.isin(fold_of_row, list(val_folds))].reset_index(drop=True)
    df_test = df[np.isin(fold_of_row, list(test_folds))].reset_index(drop=True)

    print(f"      Train : {len(df_train)}")
    print(f"      Val   : {len(df_val)}")
    print(f"      Test  : {len(df_test)}")

    # Sanity check: no group (physical deal) should appear in more than
    # one split.
    g_train = set((df_train["_source_file"].astype(str) + "||" + df_train["_board_number"].astype(str)))
    g_val = set((df_val["_source_file"].astype(str) + "||" + df_val["_board_number"].astype(str)))
    g_test = set((df_test["_source_file"].astype(str) + "||" + df_test["_board_number"].astype(str)))
    overlap = (g_train & g_val) | (g_train & g_test) | (g_val & g_test)
    if overlap:
        raise RuntimeError(f"Group leakage across splits: {len(overlap)} group(s) appear in >1 split")
    print("      No cross-split group leakage (verified).")

    missing_classes = set(df["label"]) - set(df_train["label"])
    if missing_classes:
        names = [le.inverse_transform([c])[0] for c in missing_classes]
        print(f"[WARN] {len(missing_classes)} class(es) absent from train after group-aware split: {names}")

    # Save splits
    splits = {"train": df_train, "val": df_val, "test": df_test, "full": df}
    for name, split_df in splits.items():
        path = output_dir / f"{name}.csv"
        split_df.to_csv(path, index=False)
        print(f"      Saved: {path}")

    # Save feature column list
    feat_path = output_dir / "feature_columns.json"
    feat_path.write_text(json.dumps(feature_cols, indent=2))
    print(f"      Feature cols saved: {feat_path}")

    return splits


def load_splits(
    processed_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], LabelEncoder]:
    """
    Load previously built train/val/test splits and supporting artifacts.

    Returns:
        df_train, df_val, df_test, feature_cols, label_encoder
    """
    processed_dir = Path(processed_dir)

    df_train = pd.read_csv(processed_dir / "train.csv")
    df_val = pd.read_csv(processed_dir / "val.csv")
    df_test = pd.read_csv(processed_dir / "test.csv")

    feature_cols: list[str] = json.loads((processed_dir / "feature_columns.json").read_text())

    with open(processed_dir / "label_encoder.pkl", "rb") as fp:
        le: LabelEncoder = pickle.load(fp)

    return df_train, df_val, df_test, feature_cols, le
