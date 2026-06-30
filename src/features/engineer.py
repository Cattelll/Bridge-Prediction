"""
Feature engineering for bridge contract prediction.

Extracts hand-based, deal-level, and auction-based features
from a BoardRecord into a flat dictionary suitable for ML.

Feature groups:
  hand_*     : per-seat features  (suffix: _N, _E, _S, _W)
  ns_* / ew_*: partnership features
  deal_*     : dealer + vulnerability
  auction_*  : derived from the bidding sequence
  target     : contract label (e.g. "3N", "4S", "PASS")
  target_cat : contract category (Partscore / Game / SmallSlam / GrandSlam / Pass)
"""

from __future__ import annotations

from typing import Optional

from src.parser.lin_parser import BoardRecord, Hand


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HCP_VALUES: dict[str, int] = {"A": 4, "K": 3, "Q": 2, "J": 1}
CONTROL_VALUES: dict[str, int] = {"A": 2, "K": 1}
RANK_ORDER: dict[str, int] = {r: i for i, r in enumerate("AKQJT98765432")}

SUITS: list[str] = ["S", "H", "D", "C"]
SEATS: list[str] = ["N", "E", "S", "W"]

# All valid contract strings in ascending order
_STRAINS_ORDER = ["C", "D", "H", "S", "N"]
CONTRACT_LABELS: list[str] = ["PASS"] + [
    f"{level}{strain}"
    for level in range(1, 8)
    for strain in _STRAINS_ORDER
]  # 36 classes: PASS + 35 contracts

GAME_CONTRACTS: set[str] = {"3N", "4S", "4H", "5D", "5C"}

VULNERABILITY_LABELS: list[str] = ["None", "NS", "EW", "Both"]
DEALER_LABELS: list[str] = ["N", "E", "S", "W"]


# ---------------------------------------------------------------------------
# Single-hand features
# ---------------------------------------------------------------------------

def _suit_cards(hand: Hand, suit: str) -> list[str]:
    mapping = {"S": hand.spades, "H": hand.hearts, "D": hand.diamonds, "C": hand.clubs}
    return mapping[suit]


def hand_hcp_total(hand: Hand) -> int:
    return sum(HCP_VALUES.get(c, 0) for suit in SUITS for c in _suit_cards(hand, suit))


def hand_hcp_per_suit(hand: Hand) -> dict[str, int]:
    return {
        suit: sum(HCP_VALUES.get(c, 0) for c in _suit_cards(hand, suit))
        for suit in SUITS
    }


def hand_suit_lengths(hand: Hand) -> dict[str, int]:
    return {suit: len(_suit_cards(hand, suit)) for suit in SUITS}


def hand_controls(hand: Hand) -> int:
    """A=2, K=1 controls across all suits."""
    return sum(CONTROL_VALUES.get(c, 0) for suit in SUITS for c in _suit_cards(hand, suit))


def hand_is_balanced(hand: Hand) -> bool:
    """4-3-3-3, 4-4-3-2, or 5-3-3-2 distributions."""
    lengths = sorted(len(_suit_cards(hand, s)) for s in SUITS)
    return lengths in [
        [3, 3, 3, 4],
        [2, 3, 4, 4],
        [2, 3, 3, 5],
    ]


def hand_void_count(hand: Hand) -> int:
    return sum(1 for s in SUITS if len(_suit_cards(hand, s)) == 0)


def hand_singleton_count(hand: Hand) -> int:
    return sum(1 for s in SUITS if len(_suit_cards(hand, s)) == 1)


def hand_doubleton_count(hand: Hand) -> int:
    return sum(1 for s in SUITS if len(_suit_cards(hand, s)) == 2)


def hand_ltc(hand: Hand) -> int:
    """Losing Trick Count (LTC).

    Per suit: count = min(suit_length, 3), then subtract A/K/Q present
    in the top min(suit_length, 3) positions.
    """
    total = 0
    for suit in SUITS:
        cards = _suit_cards(hand, suit)
        n = len(cards)
        if n == 0:
            continue
        count = min(n, 3)
        top_cards = sorted(cards, key=lambda r: RANK_ORDER.get(r, 99))[:count]
        honors = sum(1 for c in top_cards if c in ("A", "K", "Q"))
        total += count - honors
    return total


def hand_stopper_per_suit(hand: Hand) -> dict[str, bool]:
    """True if the hand has a stopper in the given suit.

    Stopper criteria: A | Kx+ | Qxx+ | Jxxx+
    """
    stoppers: dict[str, bool] = {}
    for suit in SUITS:
        cards = _suit_cards(hand, suit)
        n = len(cards)
        card_set = set(cards)
        if "A" in card_set:
            stoppers[suit] = True
        elif "K" in card_set and n >= 2:
            stoppers[suit] = True
        elif "Q" in card_set and n >= 3:
            stoppers[suit] = True
        elif "J" in card_set and n >= 4:
            stoppers[suit] = True
        else:
            stoppers[suit] = False
    return stoppers


def hand_longest_suit(hand: Hand) -> tuple[str, int]:
    """Return (suit_letter, length) of the longest suit (ties broken: S>H>D>C)."""
    return max(
        ((s, len(_suit_cards(hand, s))) for s in SUITS),
        key=lambda x: (x[1], SUITS.index(x[0])),
    )


def hand_second_longest_suit(hand: Hand) -> tuple[str, int]:
    """Return (suit_letter, length) of the second longest suit."""
    lengths = sorted(
        ((s, len(_suit_cards(hand, s))) for s in SUITS),
        key=lambda x: (x[1], SUITS.index(x[0])),
        reverse=True,
    )
    return lengths[1] if len(lengths) > 1 else ("C", 0)


def extract_hand_features(hand: Hand, prefix: str) -> dict:
    """All per-hand features with a seat prefix (e.g. 'N_')."""
    f: dict = {}
    f[f"{prefix}hcp"] = hand_hcp_total(hand)
    for suit in SUITS:
        f[f"{prefix}hcp_{suit}"] = hand_hcp_per_suit(hand)[suit]
        f[f"{prefix}len_{suit}"] = hand_suit_lengths(hand)[suit]
        f[f"{prefix}stopper_{suit}"] = int(hand_stopper_per_suit(hand)[suit])
    f[f"{prefix}controls"] = hand_controls(hand)
    f[f"{prefix}balanced"] = int(hand_is_balanced(hand))
    f[f"{prefix}voids"] = hand_void_count(hand)
    f[f"{prefix}singletons"] = hand_singleton_count(hand)
    f[f"{prefix}doubletons"] = hand_doubleton_count(hand)
    f[f"{prefix}ltc"] = hand_ltc(hand)
    long_suit, long_len = hand_longest_suit(hand)
    f[f"{prefix}longest_suit_len"] = long_len
    # One-hot for longest suit
    for s in SUITS:
        f[f"{prefix}longest_{s}"] = int(long_suit == s)
    return f


# ---------------------------------------------------------------------------
# Partnership features
# ---------------------------------------------------------------------------

def extract_partnership_features(h1: Hand, h2: Hand, prefix: str) -> dict:
    """Combined features for two hands (e.g. NS or EW)."""
    f: dict = {}
    f[f"{prefix}hcp"] = hand_hcp_total(h1) + hand_hcp_total(h2)
    f[f"{prefix}ltc"] = hand_ltc(h1) + hand_ltc(h2)
    f[f"{prefix}controls"] = hand_controls(h1) + hand_controls(h2)

    # Fit per suit (combined length)
    combined_lens = {s: len(_suit_cards(h1, s)) + len(_suit_cards(h2, s)) for s in SUITS}
    best_fit = 0
    best_fit_suit = "S"
    for suit in SUITS:
        cl = combined_lens[suit]
        f[f"{prefix}fit_{suit}"] = cl
        f[f"{prefix}has_fit_{suit}"] = int(cl >= 8)
        if cl > best_fit:
            best_fit = cl
            best_fit_suit = suit
    f[f"{prefix}best_fit"] = best_fit
    for s in SUITS:
        f[f"{prefix}best_suit_{s}"] = int(best_fit_suit == s)

    # Stoppers per suit (either partner has it)
    s1 = hand_stopper_per_suit(h1)
    s2 = hand_stopper_per_suit(h2)
    for suit in SUITS:
        f[f"{prefix}stopper_{suit}"] = int(s1[suit] or s2[suit])

    # NT viability: stoppers in all 4 suits
    f[f"{prefix}nt_stoppers"] = int(all(s1[s] or s2[s] for s in SUITS))

    # Balanced combined
    f[f"{prefix}both_balanced"] = int(hand_is_balanced(h1) and hand_is_balanced(h2))
    return f


# ---------------------------------------------------------------------------
# Deal-level features
# ---------------------------------------------------------------------------

def extract_deal_features(board: BoardRecord) -> dict:
    f: dict = {}
    # Dealer one-hot
    for seat in DEALER_LABELS:
        f[f"dealer_{seat}"] = int(board.dealer == seat)
    # Vulnerability one-hot
    for vuln in VULNERABILITY_LABELS:
        f[f"vuln_{vuln.lower().replace(' ', '_')}"] = int(board.vulnerability == vuln)
    return f


# ---------------------------------------------------------------------------
# Auction features
# ---------------------------------------------------------------------------

def extract_auction_features(board: BoardRecord) -> dict:
    f: dict = {}
    auction = board.auction
    f["auction_len"] = len(auction)

    # Competitive: both sides made at least one non-pass bid
    SEATS_ORDER = ["N", "E", "S", "W"]
    if board.dealer in SEATS_ORDER:
        dealer_idx = SEATS_ORDER.index(board.dealer)
        ns_bids = [
            b for i, b in enumerate(auction)
            if SEATS_ORDER[(dealer_idx + i) % 4] in ("N", "S")
            and b.upper() not in ("P", "PASS", "AP", "X", "D", "DBL", "XX", "RD", "RDBL")
            and len(b) >= 2 and b[0].isdigit()
        ]
        ew_bids = [
            b for i, b in enumerate(auction)
            if SEATS_ORDER[(dealer_idx + i) % 4] in ("E", "W")
            and b.upper() not in ("P", "PASS", "AP", "X", "D", "DBL", "XX", "RD", "RDBL")
            and len(b) >= 2 and b[0].isdigit()
        ]
    else:
        ns_bids, ew_bids = [], []

    f["auction_ns_bids"] = len(ns_bids)
    f["auction_ew_bids"] = len(ew_bids)
    f["auction_competitive"] = int(len(ns_bids) > 0 and len(ew_bids) > 0)
    f["auction_has_double"] = int(board.doubled or board.redoubled)
    f["auction_doubled"] = int(board.doubled)
    f["auction_redoubled"] = int(board.redoubled)

    # Opening bid (first non-pass bid)
    opening_bid: Optional[str] = None
    for bid in auction:
        b = bid.upper().strip()
        if b not in ("P", "PASS", "AP"):
            opening_bid = b
            break

    if opening_bid and len(opening_bid) >= 2 and opening_bid[0].isdigit():
        f["opening_level"] = int(opening_bid[0])
        raw_strain = opening_bid[1:].upper().replace("NT", "N")
        from src.parser.lin_parser import _STRAIN_NORM
        opening_strain = _STRAIN_NORM.get(raw_strain, raw_strain)
    else:
        f["opening_level"] = 0
        opening_strain = "PASS"

    for s in ["C", "D", "H", "S", "N", "PASS"]:
        f[f"opening_strain_{s}"] = int(opening_strain == s)

    # Number of alerts in auction
    f["auction_alerts"] = len(board.alerts)

    return f


# ---------------------------------------------------------------------------
# Target label
# ---------------------------------------------------------------------------

def get_contract_label(board: BoardRecord) -> str:
    """Return full contract string: 'PASS', '3N', '4S', '6Hx', etc."""
    if board.contract_level is None:
        return "PASS"
    label = f"{board.contract_level}{board.contract_strain}"
    if board.redoubled:
        label += "xx"
    elif board.doubled:
        label += "x"
    return label


def get_contract_base(board: BoardRecord) -> str:
    """Return contract without doubled marker: 'PASS', '3N', '4S' (36 classes)."""
    if board.contract_level is None:
        return "PASS"
    return f"{board.contract_level}{board.contract_strain}"


def get_contract_category(board: BoardRecord) -> str:
    """Return: Pass / Partscore / Game / SmallSlam / GrandSlam."""
    if board.contract_level is None:
        return "Pass"
    if board.contract_level == 7:
        return "GrandSlam"
    if board.contract_level == 6:
        return "SmallSlam"
    base = f"{board.contract_level}{board.contract_strain}"
    if base in GAME_CONTRACTS:
        return "Game"
    return "Partscore"


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_features(board: BoardRecord) -> Optional[dict]:
    """
    Extract all features from a BoardRecord.

    Returns None if the board has incomplete hands (< 4 seats with 13 cards).
    """
    # Validate completeness
    if len(board.hands) < 4:
        return None
    if any(board.hands.get(s) is None or board.hands[s].total_cards() != 13 for s in SEATS):
        return None

    f: dict = {}

    # Per-seat hand features
    for seat in SEATS:
        f.update(extract_hand_features(board.hands[seat], prefix=f"{seat}_"))

    # Partnership features
    f.update(extract_partnership_features(board.hands["N"], board.hands["S"], prefix="ns_"))
    f.update(extract_partnership_features(board.hands["E"], board.hands["W"], prefix="ew_"))

    # HCP advantage
    f["hcp_ns_advantage"] = f["ns_hcp"] - f["ew_hcp"]

    # Deal-level
    f.update(extract_deal_features(board))

    # Auction
    f.update(extract_auction_features(board))

    # Metadata (not used as ML features, kept for reference)
    f["_board_number"] = board.board_number
    f["_source_file"] = board.source_file
    f["_room"] = board.room
    f["_declarer"] = board.declarer
    f["_result"] = board.result
    f["_tricks_made"] = board.tricks_made

    # Targets
    f["target"] = get_contract_label(board)          # full label incl. doubled (66 classes)
    f["target_base"] = get_contract_base(board)      # base label without doubled (36 classes)
    f["target_category"] = get_contract_category(board)  # 5 categories

    return f
