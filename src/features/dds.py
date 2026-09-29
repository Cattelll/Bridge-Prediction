"""
Double-Dummy Solver (DDS) derived features — optional, see CLAUDE.md scope.

Uses `endplay` (Python wrapper for Bo Haglund's DDS) to compute, per board,
the exact number of tricks each partnership can take in each strain with
perfect play (full information), and the theoretical par contract. Import
of `endplay` is lazy (only on first call) so the base pipeline does not
require it unless DDS features are explicitly requested.
"""

from __future__ import annotations

from typing import Optional

from src.parser.lin_parser import BoardRecord

SEATS_ORDER = ["N", "E", "S", "W"]
STRAINS = ["S", "H", "D", "C", "N"]

# Order MUST match the column order produced by compute_dds_features() +
# expand_par_denom() (ns/ew interleaved per strain, then par scalars, then the
# one-hot par denom) — this is the order that lands in
# data/processed/feature_columns.json. Do not "tidy" it into ns-block / ew-block.
DDS_FEATURE_COLUMNS: list[str] = (
    [col for s in STRAINS for col in (f"ns_dd_{s}", f"ew_dd_{s}")]
    + ["dd_par_level", "dd_par_declarer_is_ns", "dd_par_score"]
    + [f"dd_par_denom_{s}" for s in STRAINS]
)


def par_contract_key(contract: str, declarer: str) -> str:
    """Canonical key for comparing a bid contract with a DDS par contract.

    `contract` is in the `target` column format ("PASS", "4S", "5Dx", "3Nxx");
    `declarer` is any seat letter (N/E/S/W) — only the side (NS/EW) matters.
    A pass-out has no declarer, so its key is just "PASS". Returns "" when
    the declarer side is unknown, so it never matches a par contract.
    """
    if contract == "PASS":
        return "PASS"
    seat = "".join(ch for ch in str(declarer).upper() if ch in "NESW")
    if not seat:
        return ""
    return f"{contract}@{'NS' if seat in ('N', 'S') else 'EW'}"


def _board_to_pbn(board: BoardRecord) -> str:
    parts = []
    for seat in SEATS_ORDER:
        h = board.hands[seat]
        parts.append(".".join("".join(suit) for suit in [h.spades, h.hearts, h.diamonds, h.clubs]))
    return "N:" + " ".join(parts)


def _is_complete_deal(board: BoardRecord) -> bool:
    """True if the board is a real 4x13, 52-distinct-card deal."""
    if len(board.hands) < 4 or any(
        board.hands.get(s) is None or board.hands[s].total_cards() != 13 for s in SEATS_ORDER
    ):
        return False

    # Some manually-transcribed archives (e.g. 1990s-2000s PBN exports) have
    # the same card listed in two hands and another card missing entirely —
    # each hand still reports 13 cards, but the deck isn't a real 52-card
    # deck. endplay's DDS solver segfaults (not a catchable exception) on
    # such input, so reject it before calling into the C library.
    seen_cards: set[tuple[str, str]] = set()
    for seat in SEATS_ORDER:
        h = board.hands[seat]
        for suit, cards in (("S", h.spades), ("H", h.hearts), ("D", h.diamonds), ("C", h.clubs)):
            for rank in cards:
                key = (suit, rank.upper())
                if key in seen_cards:
                    return False
                seen_cards.add(key)
    return len(seen_cards) == 52


def _features_from_table(board: BoardRecord, table) -> dict:
    """DD trick counts + par contract(s) for one board, from its DD table."""
    from endplay.dds import par
    from endplay.types import Vul, Player, Denom

    denom_for_strain = {
        "S": Denom.spades, "H": Denom.hearts, "D": Denom.diamonds,
        "C": Denom.clubs, "N": Denom.nt,
    }
    vul_map = {"None": Vul.none, "NS": Vul.ns, "EW": Vul.ew, "Both": Vul.both}
    dealer_to_player = {"N": Player.north, "E": Player.east, "S": Player.south, "W": Player.west}

    row: dict = {}
    for strain in STRAINS:
        denom = denom_for_strain[strain]
        row[f"ns_dd_{strain}"] = table[Player.north, denom]
        row[f"ew_dd_{strain}"] = table[Player.east, denom]

    vul = vul_map.get(board.vulnerability, Vul.none)
    dealer_player = dealer_to_player.get(board.dealer, Player.north)
    par_list = par(table, vul, dealer_player)
    contracts = list(par_list)

    # A board can have several par contracts with the same par score
    # (e.g. 3NT by N or 5C by S; 5Cx or 5Dx sacrifice) — keep ALL of
    # them in dd_par_contracts, see par_contract_key(). The scalar
    # dd_par_* columns still describe only the first one.
    if not contracts:  # par is a pass-out
        row["dd_par_level"] = 0
        row["dd_par_denom"] = -1
        row["dd_par_declarer_is_ns"] = 0
        row["dd_par_score"] = 0
        row["dd_par_contracts"] = "PASS"
        return row

    first = contracts[0]
    row["dd_par_level"] = int(first.level)
    row["dd_par_denom"] = int(first.denom)
    row["dd_par_declarer_is_ns"] = int(first.declarer in (Player.north, Player.south))
    row["dd_par_score"] = int(par_list.score)
    penalty_suffix = {"passed": "", "undoubled": "", "doubled": "x", "redoubled": "xx"}
    keys = {
        par_contract_key(
            f"{int(c.level)}{STRAINS[int(c.denom)]}{penalty_suffix[c.penalty.name]}",
            "N" if c.declarer in (Player.north, Player.south) else "E",
        )
        for c in contracts
    }
    row["dd_par_contracts"] = "|".join(sorted(keys))
    return row


def compute_dds_features(board: BoardRecord) -> Optional[dict]:
    """
    Compute DDS-derived features for a single (complete, 4x13) board.

    Returns None if the deal is incomplete or DDS fails on it.
    """
    if not _is_complete_deal(board):
        return None

    from endplay.dds import calc_dd_table
    from endplay.types import Deal

    try:
        return _features_from_table(board, calc_dd_table(Deal(_board_to_pbn(board))))
    except Exception:
        return None


# DDS's CalcAllTables rejects batches above its internal table limit.
_BATCH_SIZE = 32


def compute_dds_features_batch(boards: list[BoardRecord]) -> list[Optional[dict]]:
    """Same result as [compute_dds_features(b) for b in boards], ~3-4x faster:
    DD tables are solved in batches via DDS's own internal threading
    (calc_all_tables — no Python multiprocessing). A failing batch falls back
    to solving its boards one by one."""
    from endplay.dds import calc_all_tables
    from endplay.types import Deal

    results: list[Optional[dict]] = [None] * len(boards)
    valid = [i for i, b in enumerate(boards) if _is_complete_deal(b)]
    for start in range(0, len(valid), _BATCH_SIZE):
        idx = valid[start:start + _BATCH_SIZE]
        try:
            tables = list(calc_all_tables([Deal(_board_to_pbn(boards[i])) for i in idx]))
        except Exception:
            for i in idx:
                results[i] = compute_dds_features(boards[i])
            continue
        for i, table in zip(idx, tables):
            try:
                results[i] = _features_from_table(boards[i], table)
            except Exception:
                results[i] = None
    return results


def expand_par_denom(df):
    """One-hot encode the raw `dd_par_denom` int column (0-4) into
    `dd_par_denom_S/H/D/C/N`, matching `DDS_FEATURE_COLUMNS`. Safe to call on
    a DataFrame that already has the raw column, whether it came from a
    freshly computed DDS dict or a cached CSV using the same raw schema."""
    df = df.copy()
    for i, strain in enumerate(STRAINS):
        df[f"dd_par_denom_{strain}"] = (df["dd_par_denom"] == i).astype(int)
    return df.drop(columns=["dd_par_denom"])
