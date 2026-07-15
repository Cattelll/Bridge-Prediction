import sys
import time
from pathlib import Path

import pandas as pd
from endplay.dds import calc_dd_table, par
from endplay.types import Deal, Vul, Player, Denom

PROJECT_ROOT = Path("d:/Bridge-Prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import LINParser
from src.features.engineer import extract_features

SEATS_ORDER = ["N", "E", "S", "W"]
STRAINS = ["S", "H", "D", "C", "N"]
DENOM_FOR_STRAIN = {
    "S": Denom.spades, "H": Denom.hearts, "D": Denom.diamonds,
    "C": Denom.clubs, "N": Denom.nt,
}
VUL_MAP = {"None": Vul.none, "NS": Vul.ns, "EW": Vul.ew, "Both": Vul.both}
DEALER_TO_PLAYER = {"N": Player.north, "E": Player.east, "S": Player.south, "W": Player.west}


def board_to_pbn(board) -> str:
    parts = []
    for seat in SEATS_ORDER:
        h = board.hands[seat]
        parts.append(".".join("".join(suit) for suit in [h.spades, h.hearts, h.diamonds, h.clubs]))
    return "N:" + " ".join(parts)


def main():
    t_start = time.time()
    parser = LINParser()
    boards = parser.parse_directory(PROJECT_ROOT / "data" / "raw")
    print(f"Parsed {len(boards)} boards", flush=True)

    rows = []
    n_ok = 0
    n_skip = 0
    for i, b in enumerate(boards):
        if len(b.hands) < 4 or any(
            b.hands.get(s) is None or b.hands[s].total_cards() != 13 for s in SEATS_ORDER
        ):
            n_skip += 1
            continue

        base_row = extract_features(b)
        if base_row is None:
            n_skip += 1
            continue

        try:
            pbn = board_to_pbn(b)
            deal = Deal(pbn)
            table = calc_dd_table(deal)

            row = {
                "_source_file": b.source_file,
                "_room": b.room,
                "_board_number": b.board_number,
            }
            for strain in STRAINS:
                denom = DENOM_FOR_STRAIN[strain]
                row[f"ns_dd_{strain}"] = table[Player.north, denom]
                row[f"ew_dd_{strain}"] = table[Player.east, denom]

            vul = VUL_MAP.get(b.vulnerability, Vul.none)
            dealer_player = DEALER_TO_PLAYER.get(b.dealer, Player.north)
            par_list = par(table, vul, dealer_player)
            first = list(par_list)[0]
            row["dd_par_level"] = int(first.level)
            row["dd_par_denom"] = int(first.denom)
            row["dd_par_declarer_is_ns"] = int(first.declarer in (Player.north, Player.south))
            row["dd_par_score"] = int(par_list.score)

            rows.append(row)
            n_ok += 1
        except Exception as e:
            n_skip += 1
            print(f"  [WARN] skip board {b.source_file}/{b.room}/{b.board_number}: {e}", flush=True)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            print(f"  progress: {i+1}/{len(boards)} parsed, {n_ok} DDS computed, "
                  f"{elapsed:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "experiments" / "2026-07-15" / "outputs" / "dds_cache.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t_start
    print(f"Done: {n_ok} boards with DDS, {n_skip} skipped, {elapsed:.0f}s total", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
