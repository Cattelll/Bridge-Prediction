"""
PBN (Portable Bridge Notation) parser — produces the same BoardRecord/Hand
objects as LINParser, so downstream code (extract_features, build_dataset)
works unchanged regardless of source format.

Used for supplementary data sources (e.g. computerbridge.se championship
archives) that publish deals in PBN rather than BBO's LIN format. See
CLAUDE.md / experiments/2026-07-15/ for provenance and scope notes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .lin_parser import BoardRecord, Hand, _STRAIN_NORM

_SEAT_ROTATION = ["N", "E", "S", "W"]
_TAG_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
_RANK_ORDER = {r: i for i, r in enumerate("AKQJT98765432")}


def _parse_deal_field(deal_str: str) -> dict[str, Hand]:
    """Parse e.g. 'N:QJT85.9.J.J87652 A.A642.T9432.AK9 ...' into 4 Hands,
    keyed by seat letter, honoring whichever seat the string starts from."""
    start_seat, _, rest = deal_str.partition(":")
    start_seat = start_seat.strip().upper()
    hand_strs = rest.strip().split()
    if start_seat not in _SEAT_ROTATION or len(hand_strs) != 4:
        return {}

    start_idx = _SEAT_ROTATION.index(start_seat)
    hands: dict[str, Hand] = {}
    for offset, hand_str in enumerate(hand_strs):
        seat = _SEAT_ROTATION[(start_idx + offset) % 4]
        suits = hand_str.split(".")
        if len(suits) != 4:
            continue
        hand = Hand()
        hand.spades = list(suits[0])
        hand.hearts = list(suits[1])
        hand.diamonds = list(suits[2])
        hand.clubs = list(suits[3])
        for attr in ("spades", "hearts", "diamonds", "clubs"):
            getattr(hand, attr).sort(key=lambda r: _RANK_ORDER.get(r.upper(), 99))
        hands[seat] = hand
    return hands


def _parse_contract(contract_str: str) -> tuple[Optional[int], str, bool, bool]:
    """Parse e.g. '4SX' -> (4, 'S', True, False). Returns (None, '', False, False)
    for passed-out boards."""
    contract_str = contract_str.strip()
    if not contract_str or contract_str.upper() in ("PASS", "P", "PASSED OUT"):
        return None, "", False, False

    redoubled = contract_str.upper().endswith("XX")
    doubled = not redoubled and contract_str.upper().endswith("X")
    core = contract_str[:-2] if redoubled else contract_str[:-1] if doubled else contract_str

    m = re.match(r"(\d)(NT|[SHDC])", core.upper())
    if not m:
        return None, "", False, False
    level = int(m.group(1))
    strain = _STRAIN_NORM.get(m.group(2), m.group(2))
    return level, strain, doubled, redoubled


def _parse_auction_tokens(lines: list[str]) -> list[str]:
    tokens: list[str] = []
    for line in lines:
        line = line.split("=")[0]  # drop PBN alert annotations like "2C=1"
        for tok in line.split():
            if tok in ("AP",):
                tokens.extend(["Pass", "Pass", "Pass"])
                continue
            tokens.append(tok)
    return tokens


def _parse_board_block(
    block: str, source_file: str, board_counter: int, carry: dict[str, str]
) -> Optional[BoardRecord]:
    """
    Args:
        carry: mutable dict tracking the last non-"#" value seen per PBN tag,
               since PBN uses "#" as shorthand for "same as previous board"
               (e.g. Event/HomeTeam/VisitTeam/Round repeated across a match).
               Updated in place as blocks are processed in file order.
    """
    raw_tags: dict[str, str] = {k: v for k, v in _TAG_RE.findall(block)}
    if "Deal" not in raw_tags:
        return None

    tags: dict[str, str] = {}
    for k, v in raw_tags.items():
        if v == "#" and k in carry:
            v = carry[k]
        tags[k] = v
        carry[k] = v

    hands = _parse_deal_field(tags["Deal"])
    if len(hands) != 4 or any(h.total_cards() != 13 for h in hands.values()):
        return None

    room_raw = tags.get("Room", "").strip().lower()
    room = room_raw if room_raw in ("open", "closed") else room_raw

    # Round-robin events reuse board numbers 1..N in every match of a round,
    # so the raw [Board] tag alone is NOT a unique identity within a file —
    # disambiguate by (board, home team, visiting team). HomeTeam/VisitTeam
    # are shared between a match's Open and Closed room (only Room differs),
    # so this still keeps open/closed pairs grouped together correctly.
    raw_board = tags.get("Board", str(board_counter)).strip()
    home = tags.get("HomeTeam", "").strip()
    visit = tags.get("VisitTeam", "").strip()
    if home or visit:
        board_id = f"{raw_board}|{home}|{visit}"
    else:
        # No team tags (e.g. pairs events) — round-robin board numbers can
        # still repeat across unrelated tables/sessions in the same file.
        # Fall back to the table's player names as a disambiguator. This
        # won't recognize true open/closed-room pairs as one physical deal
        # (seats are usually swapped between rooms, so names differ too) —
        # a conservative trade-off that avoids false grouping, not leakage.
        names = "|".join(sorted(tags.get(t, "") for t in ("North", "South", "East", "West")))
        board_id = f"{raw_board}|{names}"

    rec = BoardRecord(
        source_file=source_file,
        room=room,
        board_number=board_id,
        dealer=tags.get("Dealer", "").strip().upper(),
        vulnerability=tags.get("Vulnerable", "None").strip(),
        hands=hands,
    )

    # Auction: lines after the [Auction "X"] tag up to the next tag/blank/brace
    auction_match = re.search(r'\[Auction\s+"[^"]*"\]\s*\n((?:[^\[\n][^\n]*\n?)*)', block)
    if auction_match:
        auction_lines = [ln for ln in auction_match.group(1).splitlines() if ln.strip()]
        rec.auction = _parse_auction_tokens(auction_lines)

    level, strain, doubled, redoubled = _parse_contract(tags.get("Contract", ""))
    rec.contract_level = level
    rec.contract_strain = strain
    rec.doubled = doubled
    rec.redoubled = redoubled
    rec.declarer = tags.get("Declarer", "").strip().upper()

    if "Result" in tags and level is not None:
        try:
            tricks = int(tags["Result"])
            rec.tricks_made = tricks
            rec.result = tricks - (6 + level)
        except ValueError:
            pass

    for seat, tag in [("N", "North"), ("E", "East"), ("S", "South"), ("W", "West")]:
        if tag in tags:
            rec.player_names[seat] = tags[tag]

    return rec


class PBNParser:
    """Parses PBN files into BoardRecord objects (same shape as LINParser)."""

    def parse_text(self, text: str, source_file: str = "") -> list[BoardRecord]:
        blocks = re.split(r"\n\s*\n(?=\[)", text)
        records = []
        carry: dict[str, str] = {}
        for i, block in enumerate(blocks, start=1):
            if "[Deal" not in block:
                continue
            rec = _parse_board_block(block, source_file, i, carry)
            if rec is not None:
                records.append(rec)
        return records

    def parse_file(self, path: str | Path) -> list[BoardRecord]:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_text(text, source_file=path.name)

    def parse_directory(self, directory: str | Path) -> list[BoardRecord]:
        directory = Path(directory)
        records: list[BoardRecord] = []
        for path in sorted(directory.glob("*.pbn")):
            records.extend(self.parse_file(path))
        return records
