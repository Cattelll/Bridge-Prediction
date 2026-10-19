"""Build both DDS and no-DDS-feature dataset views in one shared split.

DDS is used to compute the binary matches_par_contract label. The second view
is written by build_dataset from the same rows and split assignments, with DDS
columns omitted from the model input table.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser import PBNParser  # noqa: E402
from src.preprocessing import build_dataset  # noqa: E402


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    pbn_dir = ROOT / "data" / "raw_pbn"
    dds_cache = ROOT / "data" / "combined_dds_cache_v4.csv"
    if not dds_cache.exists():
        raise FileNotFoundError(
            f"DDS v4 cache not found: {dds_cache}. It is required to make the label."
        )

    pbn_boards = PBNParser().parse_directory(pbn_dir)
    build_dataset(
        raw_dir=raw_dir,
        output_dir=ROOT / "data" / "processed",
        no_dds_output_dir=ROOT / "data" / "processed_no_dds",
        target_col="matches_par_contract",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        include_dds=True,
        dds_cache_path=dds_cache,
        extra_boards=pbn_boards,
    )


if __name__ == "__main__":
    main()
