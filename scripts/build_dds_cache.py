"""
Build the DDS feature cache used by notebooks/01_data_extraction.ipynb.

Parses data/raw/*.lin + data/raw_pbn/*.pbn exactly like notebook 01, runs
compute_dds_features() on every board, and writes one row per board keyed
by _source_file/_room/_board_number. DD tables are solved in batches with
DDS's own threading (compute_dds_features_batch) — no Python multiprocessing,
which deadlocked on this machine (see CLAUDE.md). Checkpointed every 1,000
boards so an interrupted run resumes instead of starting over.

Usage:
    python scripts/build_dds_cache.py [--out data/combined_dds_cache_v4.csv]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.dds import compute_dds_features_batch  # noqa: E402
from src.parser import LINParser, PBNParser  # noqa: E402

KEY = ["_source_file", "_room", "_board_number"]
CHECKPOINT_EVERY = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "combined_dds_cache_v4.csv"))
    args = ap.parse_args()
    out = Path(args.out)
    partial = out.with_suffix(".partial.csv")

    boards = LINParser().parse_directory(ROOT / "data" / "raw")
    boards += PBNParser().parse_directory(ROOT / "data" / "raw_pbn")
    print(f"Boards parsed: {len(boards):,}")

    rows: list[dict] = []
    done: set[tuple[str, str, str]] = set()
    if partial.exists():
        prev = pd.read_csv(partial, keep_default_na=False, na_values=[])
        rows = prev.to_dict("records")
        done = {tuple(str(r[k]) for k in KEY) for r in rows}
        print(f"Resuming from checkpoint: {len(done):,} boards already done")

    def save(path: Path) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

    # Pending boards, first occurrence of each identity key only
    pending = []
    for b in boards:
        key = (str(b.source_file), str(b.room), str(b.board_number))
        if key not in done:
            done.add(key)
            pending.append((key, b))
    print(f"Boards to solve: {len(pending):,}")

    start = time.time()
    n_failed = 0
    for c in range(0, len(pending), CHECKPOINT_EVERY):
        chunk = pending[c:c + CHECKPOINT_EVERY]
        for (key, _), row in zip(chunk, compute_dds_features_batch([b for _, b in chunk])):
            if row is None:
                n_failed += 1
                continue
            row.update(dict(zip(KEY, key)))
            rows.append(row)
        save(partial)
        print(f"  {c + len(chunk):,}/{len(pending):,} boards ({time.time() - start:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df = df[KEY + [c for c in df.columns if c not in KEY]]
    df.to_csv(out, index=False)
    if partial.exists():
        partial.unlink()
    print(f"Done: {len(df):,} rows -> {out} ({n_failed} boards skipped as invalid)")


if __name__ == "__main__":
    main()
