"""
LIN file parser for Bridge Base Online (BBO) game records.

LIN token reference:
  vg  - event/vugraph header
  pn  - player names
  rs  - results summary (all boards)
  qx  - board qualifier (o=open room, c=closed room)
  ah  - board number header
  md  - hand/dealer description
  sv  - vulnerability
  mb  - one bid in the auction
  an  - alert / announcement for the previous bid
  pc  - one card played
  mc  - claim (number of tricks taken)
  pg  - page/trick separator (ignored)
  st  - start marker (ignored)
  rh  - results header (ignored)
  nt  - notes (ignored)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Hand:
    spades: list[str] = field(default_factory=list)
    hearts: list[str] = field(default_factory=list)
    diamonds: list[str] = field(default_factory=list)
    clubs: list[str] = field(default_factory=list)

    def total_cards(self) -> int:
        return len(self.spades) + len(self.hearts) + len(self.diamonds) + len(self.clubs)

    def as_dict(self) -> dict:
        return {
            "S": self.spades,
            "H": self.hearts,
            "D": self.diamonds,
            "C": self.clubs,
        }


@dataclass
class BoardRecord:
    # Identity
    source_file: str = ""
    room: str = ""          # "open" | "closed"
    board_number: Optional[int] = None

    # Deal setup
    dealer: str = ""        # N / E / S / W
    vulnerability: str = "" # None / NS / EW / Both

    # Hands — keys are seat letters: N, E, S, W
    hands: dict[str, Hand] = field(default_factory=dict)

    # Auction
    auction: list[str] = field(default_factory=list)   # bids as strings
    alerts: dict[int, str] = field(default_factory=dict)  # bid_index -> alert text

    # Contract (derived from auction)
    contract_level: Optional[int] = None    # 1–7
    contract_strain: str = ""               # S / H / D / C / N
    declarer: str = ""                      # N / E / S / W
    doubled: bool = False
    redoubled: bool = False

    # Result
    tricks_made: Optional[int] = None       # absolute tricks taken by declarer side
    result: Optional[int] = None            # overtricks (+) or undertricks (-)

    # Play sequence
    play: list[str] = field(default_factory=list)  # card codes e.g. ["d3","dA",...]

    # Player names per seat
    player_names: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token-level utilities
# ---------------------------------------------------------------------------

# Dealer encoding in md| field: 1=S, 2=W, 3=N, 4=E
_DEALER_MAP: dict[str, str] = {"1": "S", "2": "W", "3": "N", "4": "E"}

_VULN_MAP: dict[str, str] = {
    "o": "None", "-": "None",
    "n": "NS",
    "e": "EW",
    "b": "Both",
}

# Strain normalisation: accept both upper and lower, map NT variants
_STRAIN_NORM: dict[str, str] = {
    "S": "S", "H": "H", "D": "D", "C": "C",
    "N": "N", "NT": "N",
    "s": "S", "h": "H", "d": "D", "c": "C", "n": "N",
}

_SEAT_ORDER = ["S", "W", "N", "E"]  # order of hands in md| field

_ALL_CARDS: set[str] = {
    f"{suit}{rank}"
    for suit in "SHDC"
    for rank in "AKQJT98765432"
}


def _derive_missing_hand(known_hands: dict[str, Hand]) -> Hand:
    """Infer the 4th hand from the 52-card deck minus 3 known hands."""
    used: set[str] = set()
    for hand in known_hands.values():
        for suit_letter, cards in [("S", hand.spades), ("H", hand.hearts),
                                    ("D", hand.diamonds), ("C", hand.clubs)]:
            used.update(f"{suit_letter}{c}" for c in cards)
    remaining = _ALL_CARDS - used
    derived = Hand()
    for card in remaining:
        suit_letter, rank = card[0], card[1]
        getattr(derived, {"S": "spades", "H": "hearts", "D": "diamonds", "C": "clubs"}[suit_letter]).append(rank)
    # Sort within each suit by bridge rank order
    rank_order = {r: i for i, r in enumerate("AKQJT98765432")}
    for attr in ("spades", "hearts", "diamonds", "clubs"):
        getattr(derived, attr).sort(key=lambda r: rank_order.get(r, 99))
    return derived


def _parse_hand_string(hand_str: str) -> Hand:
    """Parse e.g. 'SJ9HAQ9DQJT96CJ95' into a Hand object."""
    hand = Hand()
    current_suit: Optional[list[str]] = None
    suit_map: dict[str, list[str]] = {
        "S": hand.spades,
        "H": hand.hearts,
        "D": hand.diamonds,
        "C": hand.clubs,
    }
    i = 0
    while i < len(hand_str):
        ch = hand_str[i].upper()
        if ch in suit_map:
            current_suit = suit_map[ch]
            i += 1
        elif ch in "AKQJT98765432" and current_suit is not None:
            current_suit.append(ch)
            i += 1
        else:
            i += 1  # skip unexpected characters
    return hand


def _parse_md(md_value: str) -> tuple[str, dict[str, Hand]]:
    """
    Parse the md| field value.

    Returns (dealer, {seat: Hand}) where seat ∈ {N, E, S, W}.
    The fourth hand (East) may be absent in some LIN files and is inferred.
    """
    if not md_value:
        return "", {}

    dealer_char = md_value[0]
    dealer = _DEALER_MAP.get(dealer_char, "")
    rest = md_value[1:]

    parts = rest.split(",")
    while len(parts) < 4:
        parts.append("")

    hands: dict[str, Hand] = {}
    for seat, hand_str in zip(_SEAT_ORDER, parts):
        hands[seat] = _parse_hand_string(hand_str)

    # Derive the 4th hand if it is missing or has fewer than 13 cards
    known_seats = [s for s in _SEAT_ORDER if hands.get(s) and hands[s].total_cards() == 13]
    missing_seats = [s for s in _SEAT_ORDER if s not in known_seats]
    if len(known_seats) == 3 and len(missing_seats) == 1:
        known = {s: hands[s] for s in known_seats}
        hands[missing_seats[0]] = _derive_missing_hand(known)

    return dealer, hands


def _parse_vulnerability(sv_value: str) -> str:
    return _VULN_MAP.get(sv_value.lower().strip(), sv_value)


def _derive_contract(
    auction: list[str],
    alerts: dict[int, str],
    dealer: str,
) -> tuple[Optional[int], str, str, bool, bool]:
    """
    Derive contract details from the auction list.

    Returns (level, strain, declarer, doubled, redoubled).
    Returns (None, '', '', False, False) for passed-out boards.

    The declarer is the first player of the declaring partnership
    who bid the contract strain.
    """
    SEATS = ["N", "E", "S", "W"]
    if not dealer or dealer not in SEATS:
        return None, "", "", False, False

    dealer_idx = SEATS.index(dealer)

    last_contract_bid: Optional[str] = None
    last_contract_idx: int = -1
    doubled = False
    redoubled = False

    for i, bid in enumerate(auction):
        bid_upper = bid.upper().strip()
        if bid_upper in ("P", "PASS", "AP"):
            continue
        elif bid_upper in ("X", "D", "DBL"):
            doubled = True
            redoubled = False
        elif bid_upper in ("XX", "RD", "RDBL"):
            redoubled = True
            doubled = False
        elif len(bid_upper) >= 2 and bid_upper[0].isdigit():
            last_contract_bid = bid_upper
            last_contract_idx = i
            doubled = False
            redoubled = False

    if last_contract_bid is None:
        return None, "", "", False, False  # passed out

    level = int(last_contract_bid[0])
    strain_raw = last_contract_bid[1:].upper().replace("NT", "N")
    strain = _STRAIN_NORM.get(strain_raw, strain_raw)

    # Determine declarer: first bidder of that strain on the winning side
    winning_seat_idx = (dealer_idx + last_contract_idx) % 4
    winning_partnership = {SEATS[winning_seat_idx], SEATS[(winning_seat_idx + 2) % 4]}

    declarer = ""
    for i, bid in enumerate(auction):
        bid_upper = bid.upper().strip()
        if not (len(bid_upper) >= 2 and bid_upper[0].isdigit()):
            continue
        bid_strain = bid_upper[1:].upper().replace("NT", "N")
        bid_strain = _STRAIN_NORM.get(bid_strain, bid_strain)
        bidder_seat = SEATS[(dealer_idx + i) % 4]
        if bid_strain == strain and bidder_seat in winning_partnership:
            declarer = bidder_seat
            break

    return level, strain, declarer, doubled, redoubled


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"([a-zA-Z]{1,4})\|([^|]*)\|", re.DOTALL)


def _tokenise(text: str) -> list[tuple[str, str]]:
    """Return list of (token_name, token_value) pairs from raw LIN text."""
    return [(m.group(1).lower(), m.group(2)) for m in _TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Board-level parser
# ---------------------------------------------------------------------------

class _BoardBuilder:
    """Accumulates tokens for a single board and builds a BoardRecord."""

    def __init__(self, source_file: str, room: str) -> None:
        self.record = BoardRecord(source_file=source_file, room=room)
        self._pending_alert: bool = False

    def feed(self, token: str, value: str) -> None:
        rec = self.record
        if token == "ah":
            rec.board_number = _extract_board_number(value)
        elif token == "md":
            dealer, hands = _parse_md(value)
            rec.dealer = dealer
            rec.hands = hands
        elif token == "sv":
            rec.vulnerability = _parse_vulnerability(value)
        elif token == "mb":
            bid = value.strip().rstrip("!")  # strip alert marker
            has_alert = value.strip().endswith("!")
            rec.auction.append(bid)
            self._pending_alert = has_alert
        elif token == "an":
            # Alert belongs to the last bid
            if rec.auction:
                idx = len(rec.auction) - 1
                rec.alerts[idx] = value.strip()
            self._pending_alert = False
        elif token == "pc":
            rec.play.append(value.strip().upper())
        elif token == "mc":
            try:
                rec.tricks_made = int(value.strip())
            except ValueError:
                pass

    def build(self) -> BoardRecord:
        rec = self.record
        if rec.auction:
            level, strain, declarer, doubled, redoubled = _derive_contract(
                rec.auction, rec.alerts, rec.dealer
            )
            rec.contract_level = level
            rec.contract_strain = strain
            rec.declarer = declarer if declarer else rec.declarer
            rec.doubled = doubled
            rec.redoubled = redoubled

        if rec.tricks_made is not None and rec.contract_level is not None:
            target = rec.contract_level + 6
            rec.result = rec.tricks_made - target

        return rec


def _extract_board_number(ah_value: str) -> Optional[int]:
    """Extract integer board number from 'Board 1', '1', etc."""
    m = re.search(r"\d+", ah_value)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# File-level parser
# ---------------------------------------------------------------------------

class LINParser:
    """
    Parse one or more LIN files into a list of BoardRecord objects.

    Usage:
        parser = LINParser()
        boards = parser.parse_file("path/to/game.lin")
        all_boards = parser.parse_directory("data/raw/")
    """

    def __init__(self, encoding: str = "utf-8", fallback_encoding: str = "latin-1") -> None:
        self.encoding = encoding
        self.fallback_encoding = fallback_encoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, path: str | Path) -> list[BoardRecord]:
        path = Path(path)
        text = self._read_file(path)
        return self._parse_text(text, source_file=path.name)

    def parse_directory(
        self,
        directory: str | Path,
        pattern: str = "*.lin",
    ) -> list[BoardRecord]:
        directory = Path(directory)
        boards: list[BoardRecord] = []
        files = sorted(directory.glob(pattern))
        for f in files:
            try:
                boards.extend(self.parse_file(f))
            except Exception as exc:
                print(f"[WARN] Skipping {f.name}: {exc}")
        return boards

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            return path.read_text(encoding=self.fallback_encoding)

    def _parse_text(self, text: str, source_file: str = "") -> list[BoardRecord]:
        tokens = _tokenise(text)

        # Extract global player names from pn| (may appear multiple times)
        pn_names: list[str] = []
        for tok, val in tokens:
            if tok == "pn":
                pn_names = [n.strip() for n in val.split(",")]
                break  # use first occurrence (file-level names)

        boards: list[BoardRecord] = []
        builder: Optional[_BoardBuilder] = None

        for tok, val in tokens:
            if tok == "qx":
                # Save previous board
                if builder is not None:
                    rec = builder.build()
                    if self._is_valid(rec):
                        boards.append(rec)

                room = "open" if val.lower().startswith("o") else "closed"
                builder = _BoardBuilder(source_file=source_file, room=room)
                # Extract board number from qx value (e.g. "o13" -> 13)
                m = re.search(r"\d+", val)
                if m:
                    builder.record.board_number = int(m.group())

                # Assign seat-level player names when 8 names present
                if len(pn_names) == 8:
                    try:
                        board_local_names = [n.strip() for n in val.split(",")]
                    except Exception:
                        board_local_names = []
                    # Use file-level pn as fallback
                    if len(board_local_names) < 4:
                        board_local_names = pn_names[:4]
                    _assign_names(builder.record, board_local_names)
                elif len(pn_names) == 4:
                    _assign_names(builder.record, pn_names)

            elif tok == "pn" and builder is not None:
                # Board-specific pn| overrides the file-level one
                names = [n.strip() for n in val.split(",")]
                if len(names) >= 4:
                    _assign_names(builder.record, names[:4])

            elif builder is not None:
                builder.feed(tok, val)

        # Flush last board
        if builder is not None:
            rec = builder.build()
            if self._is_valid(rec):
                boards.append(rec)

        return boards

    @staticmethod
    def _is_valid(rec: BoardRecord) -> bool:
        """Minimal validity: must have at least a dealer and some cards."""
        return bool(rec.dealer and rec.hands)


def _assign_names(rec: BoardRecord, names: list[str]) -> None:
    """Assign player names to seats N/E/S/W from a 4-element list [S,W,N,E]."""
    seats = ["S", "W", "N", "E"]
    for seat, name in zip(seats, names):
        rec.player_names[seat] = name
