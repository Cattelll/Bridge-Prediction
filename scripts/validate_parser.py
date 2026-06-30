"""
Quick validation script for the LIN parser.
Run: py -3.12 scripts/validate_parser.py
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import LINParser, BoardRecord


def summarise(boards: list[BoardRecord]) -> None:
    total = len(boards)
    print(f"\n{'='*60}")
    print(f"Total boards parsed : {total}")

    if not boards:
        print("No boards found.")
        return

    # Passed-out boards
    passed = sum(1 for b in boards if b.contract_level is None)
    print(f"Passed-out boards   : {passed}")
    print(f"Contested boards    : {total - passed}")

    # Vulnerability distribution
    vuln_counts = Counter(b.vulnerability for b in boards)
    print(f"\nVulnerability dist  : {dict(vuln_counts)}")

    # Dealer distribution
    dealer_counts = Counter(b.dealer for b in boards)
    print(f"Dealer dist         : {dict(dealer_counts)}")

    # Contract strain distribution
    strain_counts = Counter(
        b.contract_strain for b in boards if b.contract_level is not None
    )
    print(f"Strain dist         : {dict(strain_counts)}")

    # Contract level distribution
    level_counts = Counter(
        b.contract_level for b in boards if b.contract_level is not None
    )
    print(f"Level dist          : {dict(sorted(level_counts.items()))}")

    # Doubled / redoubled
    doubled = sum(1 for b in boards if b.doubled)
    redoubled = sum(1 for b in boards if b.redoubled)
    print(f"Doubled             : {doubled}")
    print(f"Redoubled           : {redoubled}")

    # Result distribution (overtricks / undertricks)
    results = [b.result for b in boards if b.result is not None]
    if results:
        print(f"Result range        : {min(results)} to {max(results)}")

    # Boards with complete hands (all 4 seats, 13 cards each)
    complete = sum(
        1 for b in boards
        if len(b.hands) == 4 and all(h.total_cards() == 13 for h in b.hands.values())
    )
    print(f"Complete hands (4x13): {complete}/{total}")

    # Boards with play sequence
    with_play = sum(1 for b in boards if b.play)
    print(f"Boards with play seq: {with_play}")

    # Sample boards
    print(f"\n{'-'*60}")
    print("Sample boards:")
    for b in boards[:3]:
        contract_str = (
            f"{b.contract_level}{b.contract_strain}"
            + ("x" if b.doubled else "")
            + ("xx" if b.redoubled else "")
            + f"{b.declarer}"
            if b.contract_level
            else "PASS"
        )
        result_str = (
            f"{'+' if b.result and b.result > 0 else ''}{b.result}"
            if b.result is not None else "?"
        )
        print(
            f"  [{b.source_file}] Board {b.board_number} | "
            f"{b.dealer} deals | {b.vulnerability} vul | "
            f"Contract: {contract_str} {result_str} | "
            f"Bids: {len(b.auction)} | Cards: {len(b.play)}"
        )
    print(f"{'='*60}\n")
    sys.stdout.flush()


def main() -> None:
    raw_dir = Path("d:/Bridge-Prediction/data/raw")
    parser = LINParser()

    print(f"Parsing all LIN files in {raw_dir} ...")
    boards = parser.parse_directory(raw_dir)
    summarise(boards)

    # Show one full board in detail
    if boards:
        b = next((x for x in boards if x.contract_level and x.result is not None), boards[0])
        print("Detail of one board:")
        print(f"  Source     : {b.source_file}  Room: {b.room}")
        print(f"  Board #    : {b.board_number}")
        print(f"  Dealer     : {b.dealer}   Vul: {b.vulnerability}")
        print(f"  Hands:")
        for seat in ["N", "E", "S", "W"]:
            h = b.hands.get(seat)
            if h:
                print(f"    {seat}: S={h.spades} H={h.hearts} D={h.diamonds} C={h.clubs} ({h.total_cards()} cards)")
        print(f"  Auction    : {b.auction}")
        if b.alerts:
            print(f"  Alerts     : {b.alerts}")
        contract = (
            f"{b.contract_level}{b.contract_strain}"
            + ("x" if b.doubled else "")
            + ("xx" if b.redoubled else "")
            + f" by {b.declarer}"
            if b.contract_level else "PASSED OUT"
        )
        print(f"  Contract   : {contract}")
        print(f"  Tricks     : {b.tricks_made}  Result: {b.result}")
        print(f"  Play cards : {b.play[:12]}{'...' if len(b.play) > 12 else ''}")


if __name__ == "__main__":
    main()
