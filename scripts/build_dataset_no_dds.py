"""Build the official binary-label dataset without any DDS feature columns.

DDS is still read from the v4 cache to compute ``matches_par_contract``.
All DDS columns are removed before deduplication and export, leaving the
canonical hand/auction feature set. Outputs go to data/processed_no_dds/.
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import PBNParser
from src.preprocessing import build_dataset


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    pbn_dir = ROOT / "data" / "raw_pbn"
    output_dir = ROOT / "data" / "processed_no_dds"
    dds_cache = ROOT / "data" / "combined_dds_cache_v4.csv"

    if not dds_cache.exists():
        raise FileNotFoundError(
            f"DDS v4 cache not found: {dds_cache}. It is needed for the label, "
            "even though DDS feature columns are excluded from this dataset."
        )

    pbn_boards = PBNParser().parse_directory(pbn_dir)
    splits = build_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        target_col="matches_par_contract",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        include_dds=True,
        include_dds_features=False,
        dds_cache_path=dds_cache,
        extra_boards=pbn_boards,
    )

    feature_cols_path = output_dir / "feature_columns.json"
    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    for name, frame in splits.items():
        dds_cols = [
            c for c in frame.columns
            if c.startswith(("ns_dd_", "ew_dd_", "dd_par_"))
        ]
        if dds_cols:
            raise RuntimeError(f"DDS columns unexpectedly remain in {name}: {dds_cols}")
    print(f"\nNo-DDS dataset saved to: {output_dir}")
    print(f"Features: {len(feature_cols)}; full rows: {len(splits['full']):,}")
    print(
        "Positive label rate: "
        f"{splits['full']['matches_par_contract'].mean():.1%}"
    )


if __name__ == "__main__":
    main()
