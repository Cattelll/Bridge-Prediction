# Ringkasan Proyek — Bridge Contract Prediction

*Terakhir diperbarui: 2026-08-31. Sumber kebenaran naratif lengkap:
[CLAUDE.md](CLAUDE.md). Angka di sini berasal dari eksekusi ulang penuh
`notebooks/01→06` pada 2026-08-27 (commit `9a7c9ae`).*

> Catatan: [README.md](README.md) dan [docs/SUMMARY.md](docs/SUMMARY.md) masih
> mendeskripsikan pipeline lama (506 file `.lin` / 10.223 papan / 164 fitur /
> 35 kelas) dan **belum disinkronkan** ke keadaan sekarang.

---

## 1. Tujuan

Penelitian skripsi yang membandingkan tiga algoritma ensemble berbasis pohon —
**Random Forest, XGBoost, LightGBM** — untuk memprediksi **kontrak optimal**
Contract Bridge dari data kartu + lelang (rekaman BBO LIN dan arsip PBN).

- **Target utama** (`target_base`): `"PASS"` atau `"{level}{strain}"` (mis.
  `"3N"`, `"4S"`) — **36 kelas**, klasifikasi multikelas.
- Target sekunder: `target` (+ marker doubled/redoubled) dan `target_category`
  (Pass / Partscore / Game / SmallSlam / GrandSlam, 5 kelas) — hanya untuk
  analisis tambahan.

Di luar ruang lingkup (tidak dipakai, sesuai keputusan): neural network / deep
learning, reinforcement learning / game theory, prediksi real-time / streaming,
web app / API serving.

---

## 2. Dataset

| Item | Nilai |
|---|---|
| Sumber LIN | 606 file `.lin` (BBO vugraph) |
| Sumber PBN | 1.390 file `.pbn` non-BBO (43 computerbridge.se + 169 tistis.nl + 1.178 angelfire.com via Wayback Machine) |
| Board di-parse | 61.420 |
| Board setelah dedup + drop DDS | **49.755** |
| Split (train / val / test), group-aware 70/15/15 | 34.832 / 7.462 / 7.461 |
| Fitur | **182** (164 kanonik + 18 Double-Dummy Solver) |
| Kelas target (`target_base`) | 36 (PASS + 35 kontrak) |
| Random seed | 42 (tetap di semua tahap acak) |

**Disiplin provenance data.** Setiap sumber eksternal dicek: (1) legalitas —
ToS BBO eksplisit melarang scraping/automation, jadi tidak ada crawling BBO;
(2) liveness — snippet mesin pencari bukan bukti situs hidup (dua sumber
"menjanjikan" ternyata mati/di-squat); (3) kontaminasi bot — beberapa arsip PBN
mengandung bidding komputer (GIB, WBridge5) dan dikeluarkan setelah inspeksi
tag nama pemain. `angelfire.com` (mati) direcover lewat Wayback Machine.

**Split group-aware.** BBO vugraph mencatat tiap papan dua kali (open room /
closed room — dua pasangan membidik kartu **sama persis**). Split acak biasa
membiarkan ~46% pasangan itu terpecah lintas partisi → ~60% baris val/test
punya "kembaran" tangan identik di train (kebocoran: akurasi 92–95% saat
kembarannya bid kontrak sama vs 23–28% saat beda). Sekarang
`build_dataset()` memakai `StratifiedGroupKFold` dikelompokkan per papan fisik
(`_source_file` + `_board_number`) — pasangan open/closed-room selalu satu
split.

---

## 3. Fitur (182 total)

| Kelompok | Prefix | Jml | Keterangan |
|---|---|---|---|
| Per-seat hand | `N_` `E_` `S_` `W_` | 96 | HCP total & per suit, panjang suit, stopper, controls, LTC, balanced, void/singleton/doubleton |
| Partnership | `ns_` `ew_` | 44 | HCP gabungan, LTC, fit per suit, has_fit (≥8), best suit, NT stoppers |
| HCP advantage | `hcp_ns_advantage` | 1 | `ns_hcp − ew_hcp` |
| Deal context | `dealer_` `vuln_` | 8 | dealer one-hot ×4, vulnerability one-hot ×4 |
| Auction | `auction_` `opening_` | 15 | panjang lelang, competitive, bid count, doubled, opening level/strain, alerts |
| Double-Dummy Solver | `ns_dd_` `ew_dd_` `dd_par_` | 18 | trik DD per strain (NS/EW), kontrak par (level, strain, sisi, skor) |

Fitur DDS (`endplay`) disetujui masuk scope 2026-07-15 sebagai fitur tambahan,
setelah ditemukan akurasi model sudah mendekati batas konsistensi bidding
manusia (lihat Temuan §6). Semua fitur numerik/biner — tidak ada kolom string.

---

## 4. Metode / Pipeline

```
data/raw/*.lin  ─┐
                 ├─► LINParser + PBNParser ─► BoardRecord
data/raw_pbn/*.pbn┘
   ─► extract_features(board)          → 164 fitur kanonik + metadata + 3 target
   ─► compute_dds_features(board)      → 18 fitur DDS (cache: combined_dds_cache_v3.csv)
   ─► build_dataset()                  → dedup, encode label, split 70/15/15 group-aware
   ─► RFModel / XGBModel / LGBMModel .fit()   (hyperparameter dari configs/config.yaml)
   ─► evaluate()                       → accuracy, F1 macro/weighted, top-k, confusion matrix, SHAP
```

**Notebook resmi** (`notebooks/`, dieksekusi ulang penuh 2026-08-27):

| Notebook | Isi |
|---|---|
| `01_data_extraction.ipynb` | Parsing LIN + PBN + DDS → `data/processed/*.csv` |
| `02_eda_features.ipynb` | EDA & dokumentasi 182 fitur |
| `03_modeling.ipynb` | Training RF/XGBoost/LightGBM + learning curve |
| `04_evaluation.ipynb` | Evaluasi final test set, confusion matrix, SHAP, radar |
| `05_improvement_experiments.ipynb` | Eksperimen A/B/C/E/F: retuning hyperparameter + class weighting (model tunggal) |
| `06_final_evaluation.ipynb` | Laporan komprehensif 8 kandidat (3 baseline + 5 retuning) |

Semua hyperparameter dimuat dari `configs/config.yaml` (tidak di-hardcode di
notebook). RF & LightGBM memakai `class_weight="balanced"`.

---

## 5. Hasil (test set)

*Sumber: `outputs/results/test_comparison.csv` / `nb04_summary.json`.*

| Model | Accuracy | Top-3 | Top-5 | F1 Macro | F1 Weighted |
|---|---|---|---|---|---|
| RandomForest | 46,8% | 77,1% | 87,4% | 0,325 | 0,485 |
| **XGBoost** (default `config.yaml`) | **57,5%** | **82,6%** | **90,5%** | 0,390 | 0,554 |
| **LightGBM** (`class_weight="balanced"`) | 56,4% | 82,1% | 89,7% | **0,410** | **0,557** |

**Model utama proyek: LightGBM** — F1 macro tertinggi, metrik prioritas
mengingat class imbalance ekstrem (3N ~19% sampel, grand slam <1%). XGBoost
unggul di accuracy dan top-k.

### Perbandingan 8 kandidat model tunggal (`notebooks/06`, diurutkan F1 macro)

| Kandidat | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| **LightGBM (baseline resmi)** | 56,4% | **0,410** | 0,557 |
| XGBoost (Exp B: balanced penuh) | 52,4% | 0,407 | 0,540 |
| XGBoost (Exp F: soft-balanced α=0,25) | 57,5% | 0,406 | **0,560** |
| LightGBM (Exp C: retuned) | 55,6% | 0,394 | 0,544 |
| RandomForest (Exp E: retuned) | 48,9% | 0,390 | 0,507 |
| XGBoost (Exp A / baseline resmi) | 57,5% | 0,390 | 0,554 |
| RandomForest (baseline resmi) | 46,8% | 0,325 | 0,485 |

Catatan: **tidak ada kandidat yang mengalahkan LightGBM baseline di F1 macro**.
**XGBoost Exp F (soft-balanced α=0,25)** adalah "all-around terbaik" — F1 macro
nyaris menyamai LightGBM tapi unggul accuracy, F1 weighted, dan top-k; layak
jadi alternatif resmi jika prioritas condong ke accuracy.

Ensemble & stacking dari ketiga model **dicoba lalu dihapus (2026-08-31)** —
di luar ruang lingkup (perbandingan model tunggal). Sebagai catatan historis:
stacking sempat capai accuracy tertinggi 58,2% tapi F1 macro terburuk 0,384
(condong ke kelas mayoritas).

---

## 6. Temuan Utama

- **Data lebih banyak = paling efektif.** Riwayat F1 weighted: 0,481 (164
  fitur, 10.223 papan) → +DDS → +PBN (3 gelombang) → +`class_weight` →
  **0,557** (182 fitur, 49.755 papan). F1 macro naik dari ~0,29 → 0,410.
- **Batas atas struktural.** Dua pasangan expert manusia yang membidik kartu
  identik (pasangan open/closed-room BBO) hanya sepakat kontrak akhir sama
  persis **37,6%** dari waktu — model manapun yang belajar dari proxy tangan
  (HCP/shape/fit) punya plafon akurasi struktural. Akurasi ~57% sudah relatif
  dekat batas ini. Framing pelengkap: top-3/top-5 accuracy (~83% / ~90%) dan
  akurasi level kategori.
- **Fitur berpengaruh berbeda antar model.** RF & LightGBM condong ke fitur
  ringkasan lelang; XGBoost condong ke fitur kekuatan tangan (`ns_hcp`,
  `ns_has_fit_*`, `dd_par_level`) — sejalan dengan teori bidding.
- **Sirkularitas fitur lelang (temuan sekunder, belum ditindaklanjuti).**
  `auction_len`, `auction_ns_bids`, dll. dihitung dari transkrip lelang yang
  sudah selesai — ada sirkularitas konseptual. Dampaknya sedang; dicatat agar
  tidak terlewat.

---

## 7. Struktur Repo (ringkas)

```
src/
  parser/       LINParser, PBNParser, BoardRecord, Hand
  features/     engineer.py (164 fitur), dds.py (18 fitur DDS)
  preprocessing/dataset_builder.py — build_dataset(), load_splits()
  models/       RFModel, XGBModel, LGBMModel (wrapper BaseModel)
  evaluation/   evaluate(), compare_models(), save_results()
notebooks/      Pipeline resmi 01→06
notebooks_dds/  Arsip beku (identik notebooks/ pra-konsolidasi 2026-07-17)
configs/config.yaml   Hyperparameter (sumber kebenaran)
data/raw/  data/raw_pbn/       File mentah (tidak di-git)
data/processed/                CSV split + encoder (tidak di-git)
outputs/models/  outputs/results/   Model .pkl + PNG/JSON (tidak di-git)
docs/           ARCHITECTURE.md, FEATURES.md, SUMMARY.md (SUMMARY basi)
scripts/        report.py, validate_parser.py, validate_features.py,
                run_pipeline.py (BASI — jangan jalankan)
```

---

## 8. Status & Catatan

- **Notebook 01→06 konsisten & reproducible** — dieksekusi ulang penuh
  2026-08-27, semua lolos "Restart & Run All" berurutan tanpa error, output
  saling sinkron.
- **`scripts/run_pipeline.py` BASI** — masih membangun pipeline 164-fitur/
  LIN-only lama. Menjalankannya akan menimpa `data/processed/` + `outputs/models/`
  ke versi lama. Jangan jalankan sampai diupdate.
- **`README.md` dan `docs/SUMMARY.md` BASI** — belum disinkronkan ke pipeline
  49.755-papan / 182-fitur / 36-kelas.
- **`notebooks_dds/`** = arsip historis, tidak dieksekusi ulang; kini berbeda
  dari `notebooks/` (belum punya promosi XGBoost 2026-08-27).
- **Lingkungan**: Python 3.12, kernel Jupyter `bridge-venv` (`.venv`).
  `endplay` untuk DDS. Install: `pip install -e ".[notebook,dev]"`.

### Invariant (jangan ubah tanpa alasan kuat)

Random seed 42 · urutan kolom fitur (model `.pkl` terserialisasi dengan urutan
ini) · `target_base` sebagai target utama · rasio split 70/15/15 · split
group-aware per `_source_file`+`_board_number` · token handling `lin_parser.py`.

---

## 9. Riwayat Singkat

| Tanggal | Peristiwa |
|---|---|
| 2026-07-09 | Perbaikan kebocoran split → group-aware split |
| 2026-07-15 | Fitur DDS masuk scope; mulai perluasan data PBN (computerbridge.se, tistis.nl) |
| 2026-07-16 | Arsip angelfire.com via Wayback Machine (+1.178 PBN) → 49.755 papan; fix bug segfault DDS |
| 2026-07-17 | Konsolidasi: `notebooks/01–04` = satu-satunya pipeline resmi (182 fitur) |
| 2026-07-18–21 | Notebook 5 (eksperimen A–H) + Notebook 6 (evaluasi komprehensif 11 kandidat) |
| **2026-08-27** | Audit menyeluruh `notebooks/01–06` + perbaikan + eksekusi ulang penuh. XGBoost baseline dipromosikan ke default `config.yaml` (56,1% → 57,5% acc). Notebook 5 & 6 diperbaiki agar reproducible. Commit `9a7c9ae`. |
| **2026-08-31** | **Eksperimen D/G/H (ensemble + stacking) dihapus dari notebook 5 & 6 — di luar ruang lingkup. Notebook 5 = A/B/C/E/F, notebook 6 = 8 kandidat. Dijalankan ulang.** |
