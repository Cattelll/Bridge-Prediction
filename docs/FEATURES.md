# Data Dictionary — 164 Fitur

Sumber kebenaran daftar kolom: `data/processed/feature_columns.json`
(dihasilkan otomatis oleh `build_dataset()`, jangan diedit manual).
Logika perhitungan ada di `src/features/engineer.py`. Lihat
[ARCHITECTURE.md](ARCHITECTURE.md) untuk posisi tahap ini dalam pipeline.

Semua fitur numerik atau biner (0/1) — tidak ada kolom string yang masuk
ke model.

## 1. Per-Seat Hand Features (96 = 24 fitur × 4 seat)

Prefix `N_`, `E_`, `S_`, `W_`. Dihitung oleh `extract_hand_features()`.

| Fitur (tanpa prefix) | Tipe | Keterangan |
|---|---|---|
| `hcp` | int | Total High Card Points: A=4, K=3, Q=2, J=1 |
| `hcp_{S,H,D,C}` | int | HCP per suit |
| `len_{S,H,D,C}` | int | Panjang (jumlah kartu) per suit |
| `stopper_{S,H,D,C}` | 0/1 | Ada stopper di suit itu: A, atau Kx+, atau Qxx+, atau Jxxx+ |
| `controls` | int | A=2, K=1, dijumlah semua suit |
| `balanced` | 0/1 | Distribusi 4-3-3-3, 4-4-3-2, atau 5-3-3-2 |
| `voids` | int | Jumlah suit dengan 0 kartu |
| `singletons` | int | Jumlah suit dengan tepat 1 kartu |
| `doubletons` | int | Jumlah suit dengan tepat 2 kartu |
| `ltc` | int | Losing Trick Count — per suit: `min(panjang,3)` dikurangi jumlah A/K/Q di antara kartu teratas sejumlah itu |
| `longest_suit_len` | int | Panjang suit terpanjang (tie-break S>H>D>C) |
| `longest_{S,H,D,C}` | 0/1 | One-hot suit mana yang terpanjang |

## 2. Partnership Features (44 = 22 fitur × 2 partnership)

Prefix `ns_` (North-South) dan `ew_` (East-West). Dihitung oleh
`extract_partnership_features()` dari sepasang `Hand`.

| Fitur (tanpa prefix) | Tipe | Keterangan |
|---|---|---|
| `hcp` | int | Jumlah HCP kedua tangan |
| `ltc` | int | Jumlah LTC kedua tangan |
| `controls` | int | Jumlah controls kedua tangan |
| `fit_{S,H,D,C}` | int | Panjang gabungan suit (kartu partner 1 + partner 2) |
| `has_fit_{S,H,D,C}` | 0/1 | 1 jika `fit_{suit} >= 8` (fit bermain) |
| `best_fit` | int | Panjang gabungan tertinggi di antara 4 suit |
| `best_suit_{S,H,D,C}` | 0/1 | One-hot suit dengan fit terbaik |
| `stopper_{S,H,D,C}` | 0/1 | 1 jika salah satu partner punya stopper di suit itu |
| `nt_stoppers` | 0/1 | 1 jika ada stopper (gabungan) di keempat suit — indikasi NT layak |
| `both_balanced` | 0/1 | 1 jika kedua tangan balanced |

## 3. HCP Advantage (1 fitur)

| Fitur | Tipe | Keterangan |
|---|---|---|
| `hcp_ns_advantage` | int | `ns_hcp − ew_hcp` |

## 4. Deal Context (8 fitur)

Dihitung oleh `extract_deal_features()`, one-hot murni dari metadata papan.

| Fitur | Tipe | Keterangan |
|---|---|---|
| `dealer_{N,E,S,W}` | 0/1 | Siapa dealer papan ini |
| `vuln_{none,ns,ew,both}` | 0/1 | Status vulnerability |

## 5. Auction Features (15 fitur)

Dihitung oleh `extract_auction_features()` dari urutan lelang mentah (`board.auction`).

| Fitur | Tipe | Keterangan |
|---|---|---|
| `auction_len` | int | Jumlah total bid dalam lelang |
| `auction_ns_bids` | int | Jumlah bid kontrak (bukan pass/dbl) oleh NS |
| `auction_ew_bids` | int | Jumlah bid kontrak oleh EW |
| `auction_competitive` | 0/1 | 1 jika kedua sisi sama-sama pernah bid kontrak |
| `auction_has_double` | 0/1 | 1 jika kontrak akhir doubled atau redoubled |
| `auction_doubled` | 0/1 | 1 jika kontrak akhir doubled (bukan redoubled) |
| `auction_redoubled` | 0/1 | 1 jika kontrak akhir redoubled |
| `opening_level` | int | Level bid non-pass pertama (0 jika seluruh lelang pass) |
| `opening_strain_{C,D,H,S,N,PASS}` | 0/1 | One-hot strain dari opening bid |
| `auction_alerts` | int | Jumlah bid yang diberi alert/announcement |

## Metadata (bukan fitur ML)

Kolom berprefix `_` disimpan di baris fitur untuk keperluan penelusuran/debug,
tapi **dibuang** sebelum training (lihat `_META_COLS` di `dataset_builder.py`):

`_board_number`, `_source_file`, `_room`, `_declarer`, `_result`, `_tricks_made`

## Target (bukan fitur, kolom label)

Lihat `get_contract_label()`, `get_contract_base()`, `get_contract_category()`
di `src/features/engineer.py`.

| Kolom | Kelas | Contoh |
|---|---|---|
| `target_base` | 36 — **primary** | `"PASS"`, `"3N"`, `"4S"` |
| `target` | ≤66 | `"4Sx"`, `"6Nxx"` |
| `target_category` | 5 | `Pass`, `Partscore`, `Game`, `SmallSlam`, `GrandSlam` |

## Invariant

Urutan 164 kolom pada `feature_columns.json` **harus tetap sama** — model
`.pkl` di `outputs/models/` terserialisasi dengan asumsi urutan kolom input
ini. Jangan menambah/menghapus/mengurutkan ulang fitur tanpa melatih ulang
semua model.
