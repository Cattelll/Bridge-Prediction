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
_TARGET_COLS = {"target", "target_base", "target_category", "matches_par_contract"}

# DDS par columns are the direct inputs used to compute `matches_par_contract`
# (see build_dataset step 2) — keeping them as model features would leak the
# label. Kept in the saved CSVs for analysis/audit, excluded from
# feature_columns.json.
_LEAKAGE_COLS = {
    "dd_par_level", "dd_par_score", "dd_par_declarer_is_ns",
    "dd_par_denom_S", "dd_par_denom_H", "dd_par_denom_D",
    "dd_par_denom_C", "dd_par_denom_N",
}


def _get_dds_dataframe(boards, dds_cache_path: str | Path | None = None) -> pd.DataFrame:
    """Load cached DDS features if available, else compute them (slow, ~1hr/500 files).

    Both paths yield the same raw schema (`dd_par_denom` as a single 0-4 int);
    `expand_par_denom()` one-hot-encodes it uniformly regardless of source.
    """
    from src.features.dds import expand_par_denom

    if dds_cache_path is not None and Path(dds_cache_path).exists():
        print(f"      Loading cached DDS features: {dds_cache_path}")
        # keep_default_na=False: some sources have an empty "" room (no
        # open/closed distinction) — without this, "" round-trips through
        # CSV as NaN, which later .astype(str) turns into the string "nan"
        # instead of "", silently breaking the identity-key merge below.
        df_dds = pd.read_csv(dds_cache_path, keep_default_na=False, na_values=[])
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

    for col in ("_source_file", "_room", "_board_number"):
        df_dds[col] = df_dds[col].astype(str)

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
    extra_boards: list | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Parse LIN files, extract features, clean, encode, and split.

    Args:
        raw_dir        : directory containing .lin files
        output_dir     : where to save processed CSVs
        target_col     : which target to use ('target_base', 'target',
                         'target_category', or 'matches_par_contract' — the
                         last is computed from DDS par and requires
                         include_dds=True)
        train_ratio    : fraction for training
        val_ratio      : fraction for validation
        test_ratio     : fraction for test
        random_seed    : reproducibility seed
        remove_pass    : if True, drop passed-out boards from the dataset
        include_dds    : if True, add Double-Dummy Solver features (optional,
                         see CLAUDE.md scope — requires `endplay`). Also
                         required to compute the `matches_par_contract` target.
        dds_cache_path : path to a precomputed DDS CSV (keyed by
                         _source_file/_room/_board_number) to avoid recomputing;
                         if None or missing, DDS is computed from scratch
        extra_boards   : optional list of additional BoardRecord objects (e.g. from
                         PBNParser) to fold in alongside the parsed LIN boards —
                         for supplementary data sources beyond data/raw/*.lin
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

    if extra_boards:
        print(f"      + {len(extra_boards)} extra boards from supplementary source(s)")
        boards = boards + list(extra_boards)

    print("[2/5] Extracting features...")
    rows = []
    for b in boards:
        row = extract_features(b)
        if row is not None:
            rows.append(row)
    print(f"      Feature rows : {len(rows)}")

    df = pd.DataFrame(rows)

    # Identity columns can mix types across sources (e.g. LINParser gives an
    # int board_number, PBNParser gives a composite str) — normalize to str
    # everywhere so later merges/groupbys on these columns can't silently
    # miss matches due to a type mismatch (int 5 != str "5").
    for col in ("_source_file", "_room", "_board_number"):
        df[col] = df[col].astype(str)

    # ------------------------------------------------------------------
    # 2. Optional: Double-Dummy Solver features + `matches_par_contract` target
    # ------------------------------------------------------------------
    # Merged BEFORE cleaning/encoding: the `matches_par_contract` target
    # (does the actual bid contract match the DDS par contract's
    # level+strain?) depends on these columns, so they must exist before
    # target_col is cleaned/encoded below.
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

        print(f"      DDS fitur ditambahkan: {len(dds_feature_cols)}")

        # matches_par_contract: does target_base's level+strain match the DDS
        # par contract's level+strain? Built here (not in src/features/dds.py)
        # from the columns just merged above.
        from src.features.dds import STRAINS

        denom_idx = df[[f"dd_par_denom_{s}" for s in STRAINS]].to_numpy().argmax(axis=1)
        par_contract = np.where(
            df["dd_par_level"] == 0,
            "PASS",
            df["dd_par_level"].astype(int).astype(str) + np.array(STRAINS)[denom_idx],
        )
        df["matches_par_contract"] = (df["target_base"] == par_contract).astype(int)
        n_optimal = int(df["matches_par_contract"].sum())
        print(f"      matches_par_contract : {n_optimal}/{len(df)} board optimal "
              f"({n_optimal / len(df):.1%})")

    # ------------------------------------------------------------------
    # 3. Cleaning
    # ------------------------------------------------------------------
    print("[3/5] Cleaning...")

    # Optionally remove passed-out boards
    if remove_pass:
        before = len(df)
        df = df[df[target_col] != "PASS"].copy()
        print(f"      Removed {before - len(df)} passed-out boards.")

    # Feature columns = everything except metadata, target columns, and the
    # DDS par columns that directly define `matches_par_contract` (would leak
    # the label into the inputs).
    feature_cols = [
        c for c in df.columns
        if c not in _META_COLS and c not in _TARGET_COLS and c not in _LEAKAGE_COLS
    ]

    # Drop exact duplicates on feature columns only
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
    # 4. Label encoding
    # ------------------------------------------------------------------
    print("[4/5] Encoding labels...")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[target_col])
    label_map = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
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
    # 5. Train / val / test split — GROUP-AWARE by physical deal
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
