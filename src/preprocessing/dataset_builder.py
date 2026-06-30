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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.parser import LINParser
from src.features import extract_features, CONTRACT_LABELS


# Columns used as ML features (all numeric/binary — no encoding needed)
_META_COLS = {
    "_board_number", "_source_file", "_room", "_declarer",
    "_result", "_tricks_made",
}
_TARGET_COLS = {"target", "target_base", "target_category"}


def build_dataset(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    target_col: str = "target_base",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    remove_pass: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Parse LIN files, extract features, clean, encode, and split.

    Args:
        raw_dir       : directory containing .lin files
        output_dir    : where to save processed CSVs
        target_col    : which target to use ('target_base', 'target', 'target_category')
        train_ratio   : fraction for training
        val_ratio     : fraction for validation
        test_ratio    : fraction for test
        random_seed   : reproducibility seed
        remove_pass   : if True, drop passed-out boards from the dataset
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
    # 4. Train / val / test split
    # ------------------------------------------------------------------
    print("[5/5] Splitting...")

    # Stratified split (falls back to random if any class has < 2 members)
    val_test_ratio = val_ratio + test_ratio
    min_class_count = df["label"].value_counts().min()
    use_stratify = min_class_count >= 2

    if not use_stratify:
        print(f"[WARN] {(df['label'].value_counts() < 2).sum()} class(es) have <2 members "
              "— falling back to non-stratified split.")

    df_train, df_temp = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["label"] if use_stratify else None,
        random_state=random_seed,
    )
    relative_val = val_ratio / val_test_ratio

    min_temp = df_temp["label"].value_counts().min()
    use_stratify_temp = min_temp >= 2
    df_val, df_test = train_test_split(
        df_temp,
        test_size=1.0 - relative_val,
        stratify=df_temp["label"] if use_stratify_temp else None,
        random_state=random_seed,
    )

    print(f"      Train : {len(df_train)}")
    print(f"      Val   : {len(df_val)}")
    print(f"      Test  : {len(df_test)}")

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
