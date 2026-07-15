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

DDS_FEATURE_COLUMNS: list[str] = (
    [f"ns_dd_{s}" for s in STRAINS]
    + [f"ew_dd_{s}" for s in STRAINS]
    + ["dd_par_level"]
    + [f"dd_par_denom_{s}" for s in STRAINS]
    + ["dd_par_declarer_is_ns", "dd_par_score"]
)


def _board_to_pbn(board: BoardRecord) -> str:
    parts = []
    for seat in SEATS_ORDER:
        h = board.hands[seat]
        parts.append(".".join("".join(suit) for suit in [h.spades, h.hearts, h.diamonds, h.clubs]))
    return "N:" + " ".join(parts)


def compute_dds_features(board: BoardRecord) -> Optional[dict]:
    """
    Compute DDS-derived features for a single (complete, 4x13) board.

    Returns None if the deal is incomplete or DDS fails on it.
    """
    if len(board.hands) < 4 or any(
        board.hands.get(s) is None or board.hands[s].total_cards() != 13 for s in SEATS_ORDER
    ):
        return None

    from endplay.dds import calc_dd_table, par
    from endplay.types import Deal, Vul, Player, Denom

    denom_for_strain = {
        "S": Denom.spades, "H": Denom.hearts, "D": Denom.diamonds,
        "C": Denom.clubs, "N": Denom.nt,
    }
    vul_map = {"None": Vul.none, "NS": Vul.ns, "EW": Vul.ew, "Both": Vul.both}
    dealer_to_player = {"N": Player.north, "E": Player.east, "S": Player.south, "W": Player.west}

    try:
        deal = Deal(_board_to_pbn(board))
        table = calc_dd_table(deal)

        row: dict = {}
        for strain in STRAINS:
            denom = denom_for_strain[strain]
            row[f"ns_dd_{strain}"] = table[Player.north, denom]
            row[f"ew_dd_{strain}"] = table[Player.east, denom]

        vul = vul_map.get(board.vulnerability, Vul.none)
        dealer_player = dealer_to_player.get(board.dealer, Player.north)
        par_list = par(table, vul, dealer_player)
        first = list(par_list)[0]

        row["dd_par_level"] = int(first.level)
        row["dd_par_denom"] = int(first.denom)
        row["dd_par_declarer_is_ns"] = int(first.declarer in (Player.north, Player.south))
        row["dd_par_score"] = int(par_list.score)
        return row
    except Exception:
        return None


def expand_par_denom(df):
    """One-hot encode the raw `dd_par_denom` int column (0-4) into
    `dd_par_denom_S/H/D/C/N`, matching `DDS_FEATURE_COLUMNS`. Safe to call on
    a DataFrame that already has the raw column, whether it came from a
    freshly computed DDS dict or a cached CSV using the same raw schema."""
    df = df.copy()
    for i, strain in enumerate(STRAINS):
        df[f"dd_par_denom_{strain}"] = (df["dd_par_denom"] == i).astype(int)
    return df.drop(columns=["dd_par_denom"])
