"""Build a no-DDS-feature view of the official binary dataset.

DDS v4 still defines the target. This script first creates the official split
with canonical-deal grouping, then projects out every DDS column while
preserving the exact same rows and partition assignments for an ablation.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import PBNParser  # noqa: E402
from src.preprocessing import build_dataset  # noqa: E402


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    pbn_dir = ROOT / "data" / "raw_pbn"
    official_dir = ROOT / "data" / "processed"
    output_dir = ROOT / "data" / "processed_no_dds"
    dds_cache = ROOT / "data" / "combined_dds_cache_v4.csv"
    if not dds_cache.exists():
        raise FileNotFoundError(
            f"DDS v4 cache not found: {dds_cache}. It is required to make the label."
        )

    pbn_boards = PBNParser().parse_directory(pbn_dir)
    splits = build_dataset(
        raw_dir=raw_dir,
        output_dir=official_dir,
        target_col="matches_par_contract",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        include_dds=True,
        dds_cache_path=dds_cache,
        extra_boards=pbn_boards,
    )
    official_features = json.loads((official_dir / "feature_columns.json").read_text(encoding="utf-8"))
    canonical_features = [
        col for col in official_features
        if not col.startswith(("ns_dd_", "ew_dd_", "dd_par_"))
    ]
    dds_cols = [
        col for col in splits["full"].columns
        if col.startswith(("ns_dd_", "ew_dd_", "dd_par_"))
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.drop(columns=dds_cols, errors="ignore").to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "feature_columns.json").write_text(
        json.dumps(canonical_features, indent=2), encoding="utf-8"
    )
    for name in ("label_encoder.pkl", "label_map.json"):
        shutil.copy2(official_dir / name, output_dir / name)

    print(
        f"No-DDS-feature ablation data ready: {len(splits['full']):,} rows; "
        f"{len(canonical_features)} features; same canonical-deal splits as official data."
    )
    print(f"Positive rate: {splits['full']['matches_par_contract'].mean():.2%}")
    print(f"Removed DDS columns: {len(dds_cols)}")


if __name__ == "__main__":
    main()
