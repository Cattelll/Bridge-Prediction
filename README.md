# Klasifikasi Kecocokan Kontrak Par pada Contract Bridge

Penelitian skripsi yang membandingkan tiga algoritma ensemble berbasis pohon —
**Random Forest, XGBoost, LightGBM** — sebagai **classifier biner** yang
mengklasifikasikan apakah kontrak yang dibid manusia dalam Contract Bridge
**cocok dengan kontrak par** menurut Double-Dummy Solver (kelas 1) atau tidak
(kelas 0).

## Latar Belakang

Contract Bridge adalah permainan kartu yang melibatkan proses bidding (lelang)
untuk menentukan kontrak — level dan strain (suit/NT) yang akan dimainkan.
Kontrak **par** adalah kontrak "ideal" jika keempat tangan terbuka, dihitung
dengan Double-Dummy Solver (`endplay`).

**Target (sejak 2026-09-18): `matches_par_contract` — klasifikasi biner.**
Nilainya 1 jika kontrak yang benar-benar dibid cocok dengan **salah satu**
kontrak par DDS pada level, strain, status doubled/redoubled, dan sisi
declarer (NS/EW); 0 jika tidak (definisi diperketat 2026-09-27 — sebelumnya
hanya level+strain terhadap kontrak par pertama).
Sebelumnya proyek ini memakai target 36 kelas (`target_base`, mis. `"4S"`,
`"3N"`, `"PASS"`) — kolom itu tetap tersimpan di CSV sebagai data historis.

## Dataset

| Item | Nilai |
|---|---|
| Sumber | 606 file `.lin` (BBO vugraph) + 1.390 file `.pbn` (kejuaraan dunia via computerbridge.se, tistis.nl, arsip angelfire.com lewat Wayback Machine) |
| Board (setelah dedup & pembersihan) | **49.755** |
| Split train / val / test | 70/15/15 **group-aware** (34.829 / 7.462 / 7.464) |
| Fitur model | **174** (164 kanonik + 10 trik double-dummy per strain) |
| Distribusi target | 24,3% cocok par / 75,7% tidak |
| Random seed | 42 |

- **Split group-aware**: BBO vugraph mencatat tiap papan dua kali (open/closed
  room, kartu identik). Split dikelompokkan per papan fisik
  (`StratifiedGroupKFold` pada `_source_file`+`_board_number`) supaya
  pasangan itu tidak terpecah lintas split — mencegah kebocoran data.
- **Anti-leakage**: 8 kolom par DDS (`dd_par_*`) adalah bahan langsung label,
  jadi dikeluarkan dari fitur model (tetap ada di CSV untuk analisis).
- **Fitur trik DD** (`ns_dd_*`/`ew_dd_*`) dipertahankan setelah ablasi:
  +5,40pp F1 macro di validation set dibanding 164 fitur kanonik saja.

## Hasil (test set)

**Baseline resmi** (hyperparameter `configs/config.yaml`):

| Model | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | PR-AUC |
|-------|----------|----------|-------------|---------|--------|
| Random Forest | 82,3% | 0,754 | **0,821** | 0,823 | 0,667 |
| XGBoost | **83,0%** | 0,729 | 0,814 | 0,831 | 0,677 |
| **LightGBM** | 81,5% | **0,761** | 0,819 | **0,845** | **0,702** |

**Kandidat terbaik hasil retuning** (`notebooks/05`–`06`, belum dipromosikan
ke baseline): **XGBoost Eksperimen F** (`sample_weight` parsial α=0,5) —
accuracy **84,7%**, F1 macro **0,782**, F1 weighted **0,843**, ROC-AUC 0,849.

Pembanding: selalu mengklasifikasikan "tidak cocok" = 75,7% accuracy. Semua model
mengunggulinya, dan ROC-AUC ~0,82–0,85 (bukan mendekati 1,0) menunjukkan
tidak ada tanda leakage residual.

Laporan lengkap 8 kandidat: [outputs/results/REPORT.md](outputs/results/REPORT.md)
(generate ulang dengan `python scripts/report.py`).

## Struktur Proyek

```
Bridge-Prediction/
├── src/
│   ├── parser/          # LINParser (BBO) + PBNParser
│   ├── features/        # engineer.py (164 fitur kanonik), dds.py (18 kolom DDS)
│   ├── preprocessing/   # build_dataset(): label matches_par_contract, dedup, split
│   ├── models/          # RF, XGBoost, LightGBM wrappers
│   └── evaluation/      # accuracy, F1, ROC-AUC, PR-AUC, SHAP
├── notebooks/           # Pipeline resmi, jalankan berurutan 01 → 06
│   ├── 01_data_extraction.ipynb          # Parsing LIN+PBN + DDS → CSV
│   ├── 02_eda_features.ipynb             # EDA target & 174 fitur
│   ├── 03_modeling.ipynb                 # Ablasi fitur DD + training 3 model
│   ├── 04_evaluation.ipynb               # Evaluasi test set, SHAP, radar
│   ├── 05_improvement_experiments.ipynb  # Exp A/B/C/E/F (retuning, bobot kelas)
│   └── 06_final_evaluation.ipynb         # Laporan 8 kandidat
├── experiments/         # Arsip eksperimen era 36 kelas
├── scripts/             # report.py, validate_parser.py, validate_features.py
├── configs/config.yaml  # Hyperparameter & path (sumber kebenaran)
├── data/                # raw/ (.lin), raw_pbn/ (.pbn), processed/ — tidak di-git
└── outputs/             # models/ (.pkl), results/ (PNG/JSON/CSV) — tidak di-git
```

## Fitur (174 total)

| Kelompok | Jumlah | Keterangan |
|----------|--------|------------|
| Per-seat (N/E/S/W) | 96 | HCP, panjang suit, stopper, kontrol, LTC, balanced, suit terpanjang |
| Partnership (NS/EW) | 44 | HCP gabungan, fit, best suit, NT stopper |
| HCP advantage | 1 | `ns_hcp_advantage` = HCP NS − HCP EW |
| Deal context | 8 | Dealer (one-hot 4), vulnerability (one-hot 4) |
| Auction | 15 | Panjang lelang, kompetitif, opening level/strain, alert |
| Double-dummy (trik) | 10 | Trik DD per strain (S/H/D/C/N) untuk NS & EW |

## Instalasi

```powershell
git clone <repo-url>
cd Bridge-Prediction

# Python 3.12; endplay (DDS) termasuk dependency wajib
pip install -e ".[notebook,dev]"

# Daftarkan Jupyter kernel
python -m ipykernel install --user --name bridge --display-name "Python 3.12 (Bridge)"
```

## Penggunaan

### Jalankan notebook secara berurutan

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_extraction.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_features.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_improvement_experiments.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_final_evaluation.ipynb
python scripts/report.py
```

> ⚠️ **Jangan jalankan `scripts/run_pipeline.py`** — skrip itu belum diupdate
> dan masih membangun pipeline lama (LIN-only, 164 fitur, 36 kelas). Menjalankannya
> akan menimpa `data/processed/` dan `outputs/models/` ke versi usang.

### Validasi parser

```bash
python scripts/validate_parser.py
python scripts/validate_features.py
```

## Format Data

**BBO LIN** — rekaman permainan dari Bridge Base Online:

```
vg|tournament_name,...|
qx|o1|md|3SJ9HAQ9...,...,...,...|sv|0|
mb|1H|mb|P|mb|2H|mb|P|mb|4H|mb|P|mb|P|mb|P|
pc|S3|pc|S5|pc|S8|pc|SA|mc|10|
```

Token utama: `vg` (info turnamen), `md` (distribusi kartu), `sv` (vulnerability),
`mb` (bid), `pc` (kartu dimainkan), `mc` (jumlah trick).

**PBN** (Portable Bridge Notation) — format standar berbasis tag
(`[Deal "..."]`, `[Auction "..."]`, dst.), di-parse `PBNParser` menjadi
`BoardRecord` yang sama dengan LIN.

## Lingkungan

- Python 3.12
- scikit-learn ≥ 1.4, xgboost ≥ 2.0, lightgbm ≥ 4.0, endplay (DDS)
- shap ≥ 0.45, matplotlib ≥ 3.8, pandas ≥ 2.1

## Dokumentasi Tambahan

- [docs/TAHAPAN_PENELITIAN.md](docs/TAHAPAN_PENELITIAN.md) — tahapan penelitian
  berdasarkan diagram alur, dari data LIN/PBN sampai analisis hasil
- [docs/RESUME.md](docs/RESUME.md) — ringkasan dari semua dokumen ringkasan proyek
- [docs/SUMMARY.md](docs/SUMMARY.md) — ringkasan proyek satu halaman
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — alur tiap eksperimen (ablasi, Exp A/B/C/E/F)
- [docs/ALGORITHMS.md](docs/ALGORITHMS.md) — cara kerja RF vs XGBoost vs LightGBM
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — alur pipeline & tanggung jawab modul
- [docs/FEATURES.md](docs/FEATURES.md) — data dictionary fitur kanonik
- [CLAUDE.md](CLAUDE.md) — riwayat lengkap keputusan & invariant proyek

## Lisensi

Untuk keperluan penelitian/skripsi. Data BBO LIN bersumber dari Bridge Base
Online; data PBN dari arsip kejuaraan publik non-komersial.
