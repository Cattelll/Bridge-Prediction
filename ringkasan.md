# Ringkasan Proyek — Bridge Contract Prediction

*Terakhir diperbarui: 2026-09-27. Sumber kebenaran naratif lengkap:
[CLAUDE.md](CLAUDE.md). Angka di sini berasal dari eksekusi ulang penuh
`notebooks/01→06` pada 2026-09-27, setelah definisi label
`matches_par_contract` diperketat (level+strain+double+sisi declarer
terhadap semua kontrak par).*

> Catatan (2026-09-24): [README.md](README.md), [docs/SUMMARY.md](docs/SUMMARY.md),
> dan `outputs/results/REPORT.md` sudah sinkron dengan pipeline biner/174-fitur.
> Rangkuman semua dokumen ringkasan: [docs/RESUME.md](docs/RESUME.md).
> `scripts/run_pipeline.py` juga belum diupdate — jangan dijalankan (lihat §8).

---

## 1. Tujuan

Penelitian skripsi yang membandingkan tiga algoritma ensemble berbasis pohon —
**Random Forest, XGBoost, LightGBM** — untuk memprediksi apakah kontrak yang
dibid manusia **optimal** pada Contract Bridge, dari data kartu + lelang
(rekaman BBO LIN dan arsip PBN).

- **Target utama (sejak 2026-09-18): `matches_par_contract`, biner.** 1 jika
  kontrak yang benar-benar dibid (`target`, termasuk marker x/xx) dan sisi
  declarer-nya (NS/EW) cocok dengan **salah satu** kontrak **par** menurut
  Double-Dummy Solver (kolom `dd_par_contracts` — satu board bisa punya
  beberapa kontrak par berskor sama); 0 jika tidak. Sebelum 2026-09-27
  hanya level+strain `target_base` yang dibandingkan dengan kontrak par
  pertama — board yang dimainkan sisi yang salah pun dapat label 1.
- **Target historis** (dipakai sebelum 2026-09-18, tetap tersimpan di CSV
  untuk analisis, bukan lagi target training): `target_base` (36 kelas —
  "PASS" atau "{level}{strain}"), `target` (≤66 kelas dengan marker
  doubled/redoubled), `target_category` (5 kelas: Pass/Partscore/Game/
  SmallSlam/GrandSlam).

Di luar ruang lingkup (tidak dipakai, sesuai keputusan): neural network / deep
learning, reinforcement learning / game theory, prediksi real-time / streaming,
web app / API serving, ensemble/soft-voting/stacking dari ketiga model
(dicoba lalu dihapus 2026-08-31).

---

## 2. Dataset

| Item | Nilai |
|---|---|
| Sumber LIN | 606 file `.lin` (BBO vugraph) |
| Sumber PBN | 1.390 file `.pbn` non-BBO (43 computerbridge.se + 169 tistis.nl + 1.178 angelfire.com via Wayback Machine) |
| Board setelah dedup + drop DDS | **49.755** |
| Split (train / val / test), group-aware 70/15/15 | 34.829 / 7.462 / 7.464 |
| Fitur model | **174** (164 kanonik + 10 trik DD per strain) |
| Target | `matches_par_contract`, **biner** — 24,3% cocok par / 75,7% tidak |
| Random seed | 42 (tetap di semua tahap acak) |

**Anti-leakage.** Kolom par DDS (`dd_par_level`, `dd_par_denom_S/H/D/C/N`,
`dd_par_score`, `dd_par_declarer_is_ns`, `dd_par_contracts`) adalah bahan langsung pembentuk label
`matches_par_contract`, sehingga **dikeluarkan permanen** dari `feature_columns.json` —
tetap tersimpan di CSV untuk analisis/audit. 10 kolom trik DD per strain
(`ns_dd_*`/`ew_dd_*`) **dipertahankan** sebagai fitur setelah divalidasi lewat
ablasi (`notebooks/03` bagian 1b): 174 fitur mengalahkan 164 fitur kanonik
saja sebesar **+5,40pp F1 macro** di validation set (0,7496 vs 0,6956).

**Disiplin provenance data.** Setiap sumber eksternal dicek: (1) legalitas —
ToS BBO eksplisit melarang scraping/automation, jadi tidak ada crawling BBO;
(2) liveness — snippet mesin pencari bukan bukti situs hidup (dua sumber
"menjanjikan" ternyata mati/di-squat); (3) kontaminasi bot — beberapa arsip PBN
mengandung bidding komputer (GIB, WBridge5) dan dikeluarkan setelah inspeksi
tag nama pemain. `angelfire.com` (mati) direcover lewat Wayback Machine.

**Split group-aware.** BBO vugraph mencatat tiap papan dua kali (open room /
closed room — dua pasangan membidik kartu **sama persis**). Split acak biasa
membiarkan ~46% pasangan itu terpecah lintas partisi → kebocoran data.
`build_dataset()` memakai `StratifiedGroupKFold` dikelompokkan per papan fisik
(`_source_file` + `_board_number`) — pasangan open/closed-room selalu satu
split.

---

## 3. Fitur (174 total, model resmi sejak 2026-09-18)

| Kelompok | Prefix | Jml | Keterangan |
|---|---|---|---|
| Per-seat hand | `N_` `E_` `S_` `W_` | 96 | HCP total & per suit, panjang suit, stopper, controls, LTC, balanced, void/singleton/doubleton |
| Partnership | `ns_` `ew_` | 44 | HCP gabungan, LTC, fit per suit, has_fit (≥8), best suit, NT stoppers |
| HCP advantage | `ns_hcp_advantage` | 1 | `ns_hcp − ew_hcp` |
| Deal context | `dealer_` `vuln_` | 8 | dealer one-hot ×4, vulnerability one-hot ×4 |
| Auction | `auction_` `opening_` | 15 | panjang lelang, competitive, bid count, doubled, opening level/strain, alerts |
| Double-Dummy (trik) | `ns_dd_*` `ew_dd_*` | 10 | trik double-dummy per strain (S/H/D/C/N) untuk NS & EW |

164 fitur pertama = **fitur kanonik**; + 10 trik DD = **174 fitur model**.
Kolom par DDS **dikeluarkan** dari fitur (lihat §2, anti-leakage). Semua
fitur numerik/biner — tidak ada kolom string.

**Penamaan dirapikan (2026-09-20)**: `hcp_ns_advantage` → `ns_hcp_advantage`
(konsisten dengan prefix `ns_`/`ew_` kolom partnership lain); per-seat
`{seat}_longest_{S,H,D,C}` (one-hot suit terpanjang) → `{seat}_is_longest_{S,H,D,C}`
(agar tidak tertukar dengan `{seat}_len_{S,H,D,C}`, panjang suit). Kosmetik
murni — pipeline `01→06` dieksekusi ulang penuh, hasil test set identik
byte-untuk-byte dengan sebelumnya.

---

## 4. Metode / Pipeline

```
data/raw/*.lin  ─┐
                 ├─► LINParser + PBNParser ─► BoardRecord
data/raw_pbn/*.pbn┘
   ─► extract_features(board)          → 164 fitur kanonik + metadata + target historis
   ─► compute_dds_features(board)      → 18 fitur DDS + dd_par_contracts (merge di awal)
   ─► build_dataset()                  → hitung matches_par_contract (target + sisi declarer
                                          vs semua kontrak par DDS), dedup, encode label,
                                          split 70/15/15 group-aware, keluarkan kolom par
                                          DDS dari fitur (leakage)
   ─► RFModel / XGBModel / LGBMModel .fit()   (hyperparameter dari configs/config.yaml)
   ─► evaluate()                       → accuracy, F1 macro/weighted/positif, ROC-AUC, PR-AUC, SHAP
```

**Notebook resmi** (`notebooks/`, dieksekusi ulang penuh 2026-09-27):

| Notebook | Isi |
|---|---|
| `01_data_extraction.ipynb` | Parsing LIN + PBN + DDS → `data/processed/*.csv`, `target_col='matches_par_contract'` |
| `02_eda_features.ipynb` | EDA `matches_par_contract` & dokumentasi 174 fitur |
| `03_modeling.ipynb` | Ablasi fitur trik DD (1b) + training RF/XGBoost/LightGBM |
| `04_evaluation.ipynb` | Evaluasi final (`matches_par_contract`), SHAP, radar |
| `05_improvement_experiments.ipynb` | Eksperimen A/B/C/E/F: retuning hyperparameter + class weighting (model tunggal) |
| `06_final_evaluation.ipynb` | Laporan komprehensif 8 kandidat (3 baseline + 5 retuning) |

Semua hyperparameter dimuat dari `configs/config.yaml` (tidak di-hardcode di
notebook, kecuali grid manual eksperimen di nb05). RF & LightGBM memakai
`class_weight="balanced"`. Metrik top-k (tidak bermakna untuk 2 kelas) sudah
dihapus, diganti ROC-AUC/PR-AUC/precision-recall-F1 kelas positif.

---

## 5. Hasil (test set, target biner `matches_par_contract`)

*Sumber: `outputs/results/test_comparison.csv`, `nb06_final_comparison.csv`.
Baseline mayoritas (selalu tebak "tidak cocok") = 75,7% accuracy — semua
model mengungguli, dan F1 macro menegaskan model membedakan kedua kelas,
bukan sekadar menebak mayoritas. ROC-AUC ~0,82–0,85 (bukan mendekati 1,0) →
tidak ada tanda leakage residual.*

**Baseline resmi** (hyperparameter `configs/config.yaml` apa adanya):

| Model | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RandomForest | 82,3% | 0,754 | **0,821** | 0,823 | 0,667 |
| XGBoost | **83,0%** | 0,729 | 0,814 | 0,831 | 0,677 |
| **LightGBM** (`class_weight="balanced"`) | 81,5% | **0,761** | 0,819 | **0,845** | **0,702** |

**Baseline resmi terbaik: LightGBM** — F1 macro & ROC-AUC/PR-AUC tertinggi
di antara baseline, prioritas mengingat class imbalance (24,3% vs 75,7%).

### Perbandingan 8 kandidat (`notebooks/06`, 3 baseline + 5 retuning A/B/C/E/F)

| Kandidat | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Baseline RandomForest | 82,3% | 0,754 | 0,821 | 0,823 | 0,667 |
| Baseline XGBoost | 83,0% | 0,729 | 0,814 | 0,831 | 0,677 |
| Baseline LightGBM | 81,5% | 0,761 | 0,819 | 0,845 | 0,702 |
| Exp A — XGBoost retuned (500 pohon, depth 8) | **84,9%** | 0,769 | 0,839 | 0,848 | 0,709 |
| Exp B — XGBoost + sample_weight balanced penuh | 83,2% | 0,773 | 0,832 | 0,847 | 0,709 |
| Exp C — LightGBM retuned | 83,1% | 0,775 | 0,832 | **0,849** | **0,712** |
| Exp E — RandomForest retuned | 82,9% | 0,766 | 0,829 | 0,840 | 0,700 |
| **Exp F — XGBoost soft-balanced (α=0,5)** | 84,7% | **0,782** | **0,843** | **0,849** | 0,707 |

**Kesimpulan**: **XGBoost Eksperimen F (soft-balanced, `sample_weight` parsial
α=0,5)** adalah kandidat terbaik keseluruhan — F1 macro tertinggi (0,782)
SEKALIGUS accuracy tertinggi kedua (84,7%, cuma 0,2pp di bawah Exp A). Beda
dari era 36-kelas (di mana LightGBM vs XGBoost adalah trade-off tajam), untuk
`matches_par_contract` satu kandidat unggul di kedua sumbu sekaligus. **Baseline resmi
belum diubah** — promosi Exp F ke baseline belum diputuskan.

Ensemble & stacking dari ketiga model **dicoba lalu dihapus (2026-08-31)** —
di luar ruang lingkup (perbandingan model tunggal); tidak diulang untuk skema
biner.

---

## 6. Temuan Utama

- **Perubahan target ke biner mengubah trade-off model.** Era 36-kelas:
  LightGBM (F1 macro) vs XGBoost (accuracy) adalah trade-off yang jelas.
  Era `matches_par_contract` (biner): XGBoost Exp F unggul di F1 macro DAN accuracy
  sekaligus — kandidat tunggal terbaik di kedua sumbu.
  Baseline resmi tetap LightGBM (belum ada keputusan promosi Exp F).
- **Trik double-dummy per strain sangat berguna untuk label biner.** Ablasi
  174 vs 164 fitur: +5,40pp F1 macro validation — jauh di atas ambang
  keputusan (+0,5pp), dipertahankan sebagai fitur permanen.
- **Kolom par DDS = leakage, bukan noise.** Korelasi `dd_par_level`/
  `dd_par_declarer_is_ns` dengan `matches_par_contract` jauh lebih kuat dari fitur lain
  manapun (diharapkan — keduanya membentuk label) → dikeluarkan permanen dari
  fitur model, hanya dipakai untuk membentuk label itu sendiri.
- **Tidak ada tanda leakage residual pada model.** ROC-AUC ~0,82–0,85 (bukan
  mendekati 1,0) mengonfirmasi model belajar dari fitur tangan/lelang yang
  sah, bukan bocoran label.
- **Semua model jauh mengungguli baseline mayoritas** (75,7% accuracy) di
  F1 macro — bukti nyata model membedakan kedua kelas, bukan cuma menebak
  mayoritas di kelas imbalance ekstrem.
- **Seleksi fitur SHAP (2026-08-31, era 36-kelas)**: top-40 fitur (dari 182)
  setara baseline penuh, tapi gain di dalam noise — tidak dipromosikan,
  disimpan sebagai analisis untuk skripsi (`experiments/2026-08-31/`).

---

## 7. Struktur Repo (ringkas)

```
src/
  parser/       LINParser, PBNParser, BoardRecord, Hand
  features/     engineer.py (164 fitur), dds.py (18 fitur DDS + dd_par_contracts semua kontrak par)
  preprocessing/dataset_builder.py — build_dataset() (hitung matches_par_contract, keluarkan leakage), load_splits()
  models/       RFModel, XGBModel, LGBMModel (wrapper BaseModel)
  evaluation/   evaluate() (accuracy/F1/ROC-AUC/PR-AUC), compare_models(), save_results()
notebooks/      Pipeline resmi 01→06 (target biner matches_par_contract sejak 2026-09-18)
notebooks_dds/  Arsip beku — skema 36-kelas lama, tidak dieksekusi ulang
configs/config.yaml   Hyperparameter (sumber kebenaran)
data/raw/  data/raw_pbn/       File mentah (tidak di-git)
data/processed/                CSV split + encoder, 174 fitur/49.755 board (tidak di-git)
outputs/models/  outputs/results/   Model .pkl + PNG/JSON (tidak di-git)
docs/           ARCHITECTURE.md, FEATURES.md, SUMMARY.md, RESUME.md, EXPERIMENTS.md, ALGORITHMS.md
scripts/        report.py (metrik biner + 8 kandidat nb06), build_dds_cache.py, validate_parser.py,
                validate_features.py, run_pipeline.py (BASI — jangan jalankan)
```

---

## 8. Status & Catatan

- **Notebook 01→06 konsisten & reproducible** — dieksekusi ulang penuh
  2026-09-27 dengan definisi label `matches_par_contract` yang diperketat, semua lolos "Restart & Run All"
  berurutan tanpa error.
- **`scripts/run_pipeline.py` BASI** — masih membangun pipeline 164-fitur/
  LIN-only/36-kelas lama (tanpa DDS, tanpa PBN, tanpa `matches_par_contract`).
  Menjalankannya akan menimpa `data/processed/` + `outputs/models/` ke versi
  lama. Jangan jalankan sampai diupdate.
- **`scripts/report.py` sudah diupdate (2026-09-24)** — metrik biner
  (accuracy, F1 macro/weighted/positif, ROC-AUC, PR-AUC) + tabel 8 kandidat nb06;
  `REPORT.md` di-generate ulang. `README.md` ditulis ulang ke pipeline biner.
- **Masih basi**: `docs/ARCHITECTURE.md` (menyebut top-k) dan `docs/FEATURES.md`
  (10 fitur trik DD belum didokumentasikan).
- **`notebooks_dds/` dan `experiments/`** = arsip historis skema 36-kelas
  lama, tidak disentuh oleh perubahan target biner.
- **Insiden lingkungan (2026-09-18, sudah diperbaiki)**: instalasi Python 3.12
  dasar hilang dari mesin, `.venv` rusak total. Diperbaiki dengan
  `py install 3.12` + rebuild `.venv` + fix bug `build-backend` di
  `pyproject.toml` + pindahkan `endplay` (DDS) dari extra opsional ke
  dependencies wajib (kini dibutuhkan untuk label `matches_par_contract`).
- **Lingkungan**: Python 3.12, kernel Jupyter `bridge` ("Python 3.12
  (Bridge)"). `endplay` untuk DDS kini **wajib**, bukan opsional. Install:
  `pip install -e ".[notebook,dev]"`.

### Invariant (jangan ubah tanpa alasan kuat)

Random seed 42 · urutan kolom fitur (model `.pkl` terserialisasi dengan urutan
ini) · **`matches_par_contract` sebagai target utama** (sejak 2026-09-18; `target`,
`_declarer`, dan `dd_par_contracts` tetap perlu ada di CSV karena label dihitung
darinya) · kolom par DDS dikeluarkan dari `feature_columns.json` (leakage) · rasio split 70/15/15 ·
split group-aware per `_source_file`+`_board_number` · token handling
`lin_parser.py`.

---

## 9. Riwayat Singkat

| Tanggal | Peristiwa |
|---|---|
| 2026-07-09 | Perbaikan kebocoran split → group-aware split |
| 2026-07-15 | Fitur DDS masuk scope; mulai perluasan data PBN (computerbridge.se, tistis.nl) |
| 2026-07-16 | Arsip angelfire.com via Wayback Machine (+1.178 PBN) → 49.755 papan; fix bug segfault DDS |
| 2026-07-17 | Konsolidasi: `notebooks/01–04` = satu-satunya pipeline resmi (182 fitur) |
| 2026-07-18–21 | Notebook 5 (eksperimen A–H) + Notebook 6 (evaluasi komprehensif 11 kandidat) |
| 2026-08-27 | Audit menyeluruh `notebooks/01–06` + perbaikan + eksekusi ulang penuh. XGBoost baseline dipromosikan ke default `config.yaml` (56,1% → 57,5% acc, skema 36-kelas). |
| 2026-08-31 | Eksperimen D/G/H (ensemble + stacking) dihapus — di luar ruang lingkup. Eksperimen seleksi fitur SHAP (top-40 setara baseline, tidak dipromosikan). |
| 2026-09-18 | Perubahan arah penelitian: target diganti dari `target_base` (36 kelas) menjadi `matches_par_contract` (biner, vs kontrak par DDS). 8 kolom par DDS dikeluarkan dari fitur (leakage) → 174 fitur resmi. Metrik evaluasi diganti ke ROC-AUC/PR-AUC. Insiden `.venv` rusak diperbaiki. Notebook 01→06 dieksekusi ulang penuh. Model utama tetap LightGBM; kandidat terbaik keseluruhan XGBoost Exp F (soft-balanced α=0,25). |
| 2026-09-20 | Perapian nama kolom: `hcp_ns_advantage` → `ns_hcp_advantage`, `{seat}_longest_{S,H,D,C}` → `{seat}_is_longest_{S,H,D,C}` (hindari kerancuan dengan `{seat}_len_*`). Kosmetik murni di `src/features/engineer.py`; `label` (LabelEncoder atas `target_col`) dicek, ternyata load-bearing (dipakai 5 notebook + `scripts/run_pipeline.py`), tidak dihapus. Notebook 01→06 dieksekusi ulang penuh (cache DDS dipakai ulang), hasil test set identik dengan sebelumnya. |
| **2026-09-20** | **Rename kolom target `is_optimal` → `matches_par_contract` (definisi/nilai tidak berubah, nama baru menghindari istilah "optimal" yang ambigu). Diperbaiki di `dataset_builder.py`, `metrics.py`, `configs/config.yaml`, seluruh `notebooks/01–06`, `docs/SUMMARY.md`. Notebook 01→06 dieksekusi ulang penuh lagi; hasil test set identik dengan sebelumnya.** |
| **2026-09-27** | **Definisi label `matches_par_contract` diperketat: kontrak dibid (termasuk x/xx) + sisi declarer harus cocok dengan salah satu kontrak par DDS (dulu level+strain vs kontrak par pertama saja). 1.069 board 1→0, 603 board 0→1 → 24,3% positif. Cache DDS baru (`combined_dds_cache_v4.csv`, `scripts/build_dds_cache.py`). Notebook 01→06 dieksekusi ulang penuh; semua model naik (Exp F: F1 macro 0,750 → 0,782, α terpilih 0,5).** |
