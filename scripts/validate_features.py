"""
Validate feature extraction on the full dataset.
Run: py -3.12 scripts/validate_features.py
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import LINParser
from src.features import extract_features, CONTRACT_LABELS


def main() -> None:
    print("Parsing LIN files...")
    boards = LINParser().parse_directory("data/raw")
    print(f"Boards parsed: {len(boards)}")

    print("Extracting features...")
    rows = []
    skipped = 0
    for b in boards:
        row = extract_features(b)
        if row is None:
            skipped += 1
        else:
            rows.append(row)

    print(f"Feature rows  : {len(rows)}")
    print(f"Skipped       : {skipped}")

    if not rows:
        print("No rows extracted!")
        return

    # Feature count (exclude metadata and targets)
    meta_cols = {"_board_number", "_source_file", "_room", "_declarer", "_result",
                 "_tricks_made", "target", "target_category"}
    feature_cols = [k for k in rows[0] if k not in meta_cols]
    print(f"Feature count : {len(feature_cols)}")

    # Target distribution
    target_counts = Counter(r["target"] for r in rows)
    print(f"\nTop 10 contracts:")
    for label, cnt in target_counts.most_common(10):
        print(f"  {label:6s}  {cnt:5d}  ({cnt/len(rows)*100:.1f}%)")

    print(f"\nTotal unique contracts: {len(target_counts)}")
    print(f"PASS boards           : {target_counts.get('PASS', 0)}")

    cat_counts = Counter(r["target_category"] for r in rows)
    print(f"\nContract categories:")
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat:12s}  {cnt:5d}  ({cnt/len(rows)*100:.1f}%)")

    # Sample row
    print(f"\n{'='*60}")
    print("Sample feature row (first board):")
    row = rows[0]
    groups = {
        "Hand (N seat)": [k for k in feature_cols if k.startswith("N_")],
        "Partnership NS": [k for k in feature_cols if k.startswith("ns_")],
        "Deal":          [k for k in feature_cols if k.startswith("dealer_") or k.startswith("vuln_")],
        "Auction":       [k for k in feature_cols if k.startswith("auction_") or k.startswith("opening_")],
    }
    for group_name, keys in groups.items():
        print(f"\n  [{group_name}]")
        for k in keys:
            print(f"    {k:35s} = {row[k]}")

    print(f"\n  Target        : {row['target']}")
    print(f"  Target cat    : {row['target_category']}")

    # Check for None values in features
    none_counts: Counter = Counter()
    for row in rows:
        for k in feature_cols:
            if row[k] is None:
                none_counts[k] += 1
    if none_counts:
        print(f"\n[WARN] Features with None values:")
        for k, cnt in none_counts.most_common(10):
            print(f"  {k}: {cnt}")
    else:
        print(f"\n[OK] No None values in features.")


if __name__ == "__main__":
    main()
