# Ringkasan Proyek

Satu halaman untuk siapa pun yang baru masuk ke repo ini. Untuk detail lebih
dalam lihat [ARCHITECTURE.md](ARCHITECTURE.md) (alur pipeline & modul) dan
[FEATURES.md](FEATURES.md) (data dictionary fitur kanonik).

## Tujuan

Penelitian skripsi yang membandingkan tiga algoritma ensemble berbasis pohon
— Random Forest, XGBoost, LightGBM — untuk memprediksi apakah kontrak yang
benar-benar dibid (auction) dalam Contract Bridge **optimal**, dari data
hasil lelang (bidding) dan kartu, direkam dalam format BBO LIN + PBN.

> **Perubahan arah (2026-09-18)**: target diganti dari klasifikasi 36-kelas
> (`target_base`, mis. `"4S"`, `"3N"`, `"PASS"`) menjadi **klasifikasi biner
> `matches_par_contract`** — apakah level+strain kontrak yang dibid sama
> persis dengan level+strain kontrak **par** menurut Double-Dummy Solver
> (`endplay`). `target_base`/`target`/`target_category` tetap tersimpan di
> CSV sebagai kolom historis/analisis. Lihat "Status Proyek" di
> [CLAUDE.md](../CLAUDE.md) untuk riwayat lengkap perubahan ini beserta
> definisi label & anti-leakage.
>
> **Penamaan (2026-09-20)**: kolom ini sebelumnya bernama `is_optimal`,
> diganti `matches_par_contract` — nilai/definisi tidak berubah.

## Dataset

| Item | Nilai |
|---|---|
| Sumber | 606 file `.lin` (Bridge Base Online) + 1.390 file `.pbn` (non-BBO: kejuaraan dunia via computerbridge.se, tistis.nl, arsip angelfire.com) |
| Papan setelah dedup & pembersihan | 49.755 |
| Split (train / val / test, **group-aware**) | 70/15/15 |
| Fitur | 174 (164 kanonik + 10 trik double-dummy per strain — lihat [FEATURES.md](FEATURES.md)) |
| Target utama | `matches_par_contract` (biner) — 25,2% optimal / 74,8% tidak optimal |
| Random seed | 42 (tetap di semua tahap acak) |

> **Split group-aware (sejak 2026-07-09)**: BBO vugraph mencatat tiap papan
> dua kali (open room/closed room — dua pasangan membidik kartu yang **sama
> persis**). Split acak biasa membiarkan pasangan itu terpecah ke partisi
> berbeda, yang jadi sumber kebocoran data. Sekarang dikelompokkan per papan
> fisik (`StratifiedGroupKFold` pada `_source_file`+`_board_number`) di
> `build_dataset()`. Detail & angka before/after:
> [experiments/2026-07-09/README.md](../experiments/2026-07-09/README.md).

## Metode

```
LIN files  → LINParser  ─┐
PBN files  → PBNParser  ─┴→ BoardRecord
           → extract_features()        → 164 fitur kanonik + 3 kolom target historis
           → compute_dds_features()    → +18 kolom DDS (10 trik per strain + 8 kolom par)
           → build_dataset()           → hitung matches_par_contract (target_base vs par DDS),
                                          keluarkan 8 kolom par dari feature_columns.json
                                          (leakage), dedup, encode label,
                                          split 70/15/15 group-aware
           → RFModel / XGBModel / LGBMModel .fit()
           → evaluate() → accuracy, F1 macro/weighted, ROC-AUC, PR-AUC, precision/recall kelas positif
```

Fitur trik double-dummy per strain (`ns_dd_*`/`ew_dd_*`) divalidasi lewat
ablasi eksplisit (`notebooks/03_modeling.ipynb` bagian 1b) — dipertahankan
karena menaikkan F1 macro validation +4,21pp dibanding 164 fitur kanonik
saja. Detail lengkap di [ARCHITECTURE.md](ARCHITECTURE.md).

## Hasil (test set)

*Notebook 01→06 dieksekusi ulang penuh 2026-09-18 tanpa error — lihat
[outputs/results/test_comparison.csv](../outputs/results/test_comparison.csv)
dan [outputs/results/nb06_final_comparison.csv](../outputs/results/nb06_final_comparison.csv)*

**Baseline resmi** (hyperparameter `configs/config.yaml` apa adanya):

| Model | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RandomForest | 80,8% | 0,736 | 0,805 | 0,800 | 0,634 |
| XGBoost | 81,0% | 0,691 | 0,786 | 0,800 | 0,634 |
| **LightGBM** | 79,9% | **0,741** | 0,802 | **0,814** | 0,658 |

Baseline mayoritas (selalu tebak "tidak optimal", 74,8% dari test set) =
74,8% accuracy — semua model mengungguli, dan F1 macro menegaskan mereka
membedakan kedua kelas secara nyata, bukan sekadar menebak mayoritas.
ROC-AUC ~0,80–0,81 (bukan mendekati 1,0) menunjukkan tidak ada tanda
leakage residual.

**Kandidat terbaik `notebooks/05`/`06`** (retuning, belum dipromosikan ke
baseline resmi): **XGBoost Eksperimen F** (soft-balanced, `sample_weight`
parsial α=0,25) — Accuracy 82,3%, F1 Macro **0,750**, F1 Weighted 0,817,
ROC-AUC 0,818. Unggul F1 macro DAN nyaris tertinggi di accuracy sekaligus
(0,2pp di bawah kandidat accuracy-tertinggi) — beda dari trade-off tajam
yang biasa muncul di era 36-kelas lama. Perbandingan lengkap 8 kandidat (3
baseline + 5 retuning A/B/C/E/F) ada di
[outputs/results/nb06_final_comparison.csv](../outputs/results/nb06_final_comparison.csv).

## Temuan Utama

- **LightGBM (`class_weight="balanced"`) tetap model utama proyek** — F1
  macro tertinggi (prioritas utama karena class imbalance ekstrem 74,8%
  vs 25,2%).
- **XGBoost Eksperimen F (soft-balanced α=0,25)** adalah alternatif
  all-around terbaik jika prioritas condong ke accuracy/F1-weighted, tanpa
  mengorbankan F1 macro secara berarti.
- 8 kolom par DDS (`dd_par_*`) sengaja **dikeluarkan dari fitur model**
  (bukan karena tidak berguna, tapi karena jadi bahan langsung label
  `matches_par_contract` — memasukkannya sebagai fitur adalah leakage).
  Korelasinya dengan `matches_par_contract` jauh lebih kuat dibanding fitur
  lain manapun, persis seperti diharapkan dari sebuah kolom yang membentuk
  label itu sendiri.
  Lihat `notebooks/02_eda_features.ipynb` bagian 7.
- 10 fitur trik double-dummy per strain (`ns_dd_*`/`ew_dd_*`) **dipertahankan**
  — divalidasi lewat ablasi, bukan leakage karena dihitung independen dari
  kontrak par.

### Catatan sirkularitas fitur lelang (belum ditindaklanjuti)

Fitur `auction_len`, `auction_ns_bids`, `auction_ew_bids`, `auction_doubled`,
dkk. dihitung dari transkrip lelang yang **sudah selesai** — pada saat itu
kontrak final sudah ditentukan, jadi ada sirkularitas konseptual. Eksperimen
seleksi fitur SHAP (`experiments/2026-08-31/`, era 36-kelas) mengonfirmasi
kuantitatif: 3 fitur SHAP teratas justru fitur auction ini. Belum
diperbaiki/didiskusikan lebih lanjut — dicatat di sini supaya tidak terlewat.

## Batasan & Status Saat Ini

- `scripts/run_pipeline.py` **belum diupdate** — masih membangun pipeline
  164-fitur/LIN-only/36-kelas yang lama. Jangan dijalankan tanpa diupdate
  dulu mengikuti `notebooks/01_data_extraction.ipynb`, atau data & model
  resmi akan tertimpa balik ke versi usang.
- `scripts/report.py` belum diupdate untuk metrik biner — masih
  mereferensikan `top_3_accuracy` yang sudah tidak ada di
  `nb04_summary.json`/`nb06_summary.json` (tidak crash, hanya "n/a" untuk
  kolom itu).
- `notebooks_dds/` dan `experiments/` dipertahankan **tanpa perubahan**
  sebagai arsip historis skema 36-kelas lama.
- Lihat [CLAUDE.md](../CLAUDE.md) untuk batas ruang lingkup penelitian
  (algoritma/teknik apa yang sengaja tidak dipakai — termasuk ensemble/
  stacking, yang sempat dicoba lalu dihapus 2026-08-31 karena di luar
  scope) dan invariant yang tidak boleh diubah tanpa alasan kuat (seed,
  urutan fitur, target utama, rasio split, anti-leakage kolom par DDS).
