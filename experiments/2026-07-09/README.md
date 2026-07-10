# Percobaan 2026-07-09

Lima eksperimen lanjutan di luar pipeline utama (`scripts/run_pipeline.py`,
`notebooks/01-04`). Semua kode & output di sini **terpisah** dari
`outputs/models/` dan `outputs/results/` yang dipakai `scripts/report.py` —
tidak ada file kanonik yang diubah oleh eksperimen ini kecuali disebutkan
secara eksplisit dan disetujui (satu pengecualian besar: lihat
"Perbaikan kebocoran split" di bawah — itu MEMANG mengubah
`src/preprocessing/dataset_builder.py` dan `data/processed/`, dengan
persetujuan eksplisit).

Baseline pembanding untuk semua eksperimen: hasil test set di
[README.md](../../README.md) — angka terbaru (setelah perbaikan split,
lihat di bawah) XGBoost 52.1% accuracy / F1 weighted 0.481, retrain
2026-07-09/10, 506 file / 10.223 papan / 164 fitur.

## ⚠️ Perbaikan kebocoran split (2026-07-10)

Saat mengecek fitur untuk kebocoran/leakage, ditemukan masalah nyata di
**split train/val/test itu sendiri**, bukan di salah satu dari 5
eksperimen ini:

- BBO vugraph mencatat tiap papan **dua kali** — "open room" dan "closed
  room", dua pasangan membidik **kartu yang sama persis**.
- `train_test_split` acak biasa (dipakai sebelumnya) membiarkan **~46%**
  dari 4.506 pasangan itu terpecah ke split berbeda → **~60% baris
  val/test punya "kembaran" tangan identik di train.**
- Dampak terukur (XGBoost baseline, sebelum diperbaiki): akurasi **92-95%**
  pada baris yang kembarannya di train bid kontrak SAMA (hafalan), vs
  **23-28%** saat kembarannya bid kontrak BEDA (model "yakin" ke jawaban
  yang salah), vs **50.7-54.7%** pada baris tanpa kembaran sama sekali
  (estimasi generalisasi paling jujur).
- **Perbaikan**: `build_dataset()` di `src/preprocessing/dataset_builder.py`
  sekarang memakai `StratifiedGroupKFold` dikelompokkan per papan fisik
  (`_source_file` + `_board_number`), supaya pasangan open/closed-room
  selalu berada di split yang sama. Diverifikasi: 0 overlap grup
  lintas-split di split baru.
- Dataset di-rebuild & pipeline utama di-retrain 2026-07-10 di atas split
  yang diperbaiki. **Kelima notebook di bawah sudah dieksekusi ulang** di
  split baru ini (bukan sekadar catatan — angkanya benar-benar diperbarui).
- Perubahan angka ternyata **relatif kecil** (±1-2pp di level agregat,
  karena efek "hafalan" dan efek "keyakinan salah" sebagian saling
  meniadakan) — TAPI beberapa **kesimpulan** berubah, ada yang berbalik
  arah. Lihat "Ringkasan hasil" di bawah untuk detail before/after per
  eksperimen.

## Daftar percobaan

| # | Notebook | Tujuan | Status |
|---|----------|--------|--------|
| 1 | [01_hyperparameter_tuning.ipynb](01_hyperparameter_tuning.ipynb) | RandomizedSearchCV untuk RF/XGB/LGBM, bandingkan dengan default `configs/config.yaml` | done (re-run di split diperbaiki) |
| 2 | [02_smote_resampling.ipynb](02_smote_resampling.ipynb) | Atasi class imbalance dengan SMOTE / RandomOverSampler / class_weight pada train set saja | done (re-run di split diperbaiki) |
| 3 | [03_two_stage_classification.ipynb](03_two_stage_classification.ipynb) | Dua varian: (a) stage 1 PASS vs kontrak binary → stage 2 34 kelas; (b) stage 1 kategori (Pass/Partscore/Game/SmallSlam/GrandSlam) → stage 2 kontrak persis di dalam kategori | done (re-run di split diperbaiki) |
| 4 | [04_advanced_feature_engineering.ipynb](04_advanced_feature_engineering.ipynb) | Fitur tambahan dari raw hand (quick tricks, total points, Law of Total Tricks, tekanan lelang) di luar 164 fitur kanonik | done (re-run di split diperbaiki) |
| 5 | [05_advanced_boosting.ipynb](05_advanced_boosting.ipynb) | Khusus XGBoost & LightGBM: ablation sample weighting → GOSS (LGBM)/DART (XGB) → early stopping | done (re-run di split diperbaiki) |

## Aturan main

- **Split**: train/val/test dari `data/processed/` dipakai apa adanya
  (seed 42, rasio 70/15/15, **group-aware** sejak 2026-07-10 — lihat di
  atas) — tidak membuat split baru per eksperimen.
- **Val set** dipakai untuk membandingkan varian selama eksperimen; **test
  set** hanya disentuh sekali di akhir untuk kandidat yang benar-benar
  menang, supaya tidak overfit ke test set.
- **Tidak ada resampling pada val/test** — SMOTE/oversampling hanya pada
  training fold.
- Semua output eksperimen (model, JSON metrik, params terbaik) disimpan ke
  `experiments/2026-07-09/outputs/<nama_percobaan>/`.
- Kesimpulan tiap eksperimen dicatat sebagai cell markdown di akhir
  notebook masing-masing, bukan hanya di sini.

## Catatan proses & isu teknis

Dicatat supaya tidak terulang tanpa sadar di eksperimen berikutnya:

- **Kernel notebook `bridge-venv` ternyata jalan** — kernelspec-nya
  mengarah ke `D:\Bridge-Prediction\.venv\Scripts\python.exe`. CLAUDE.md
  bilang `.venv` diblokir Windows Application Control, tapi eksekusi
  `jupyter nbconvert --execute` lewat kernel ini berhasil normal di
  semua notebook — jadi blokirnya rupanya tidak berlaku (atau sudah
  tidak berlaku) untuk cara eksekusi ini. `imbalanced-learn` diinstal
  langsung ke `.venv` tersebut (dan ditambahkan ke
  `pyproject.toml`/`requirements.txt`) karena awalnya hanya ada di
  Python sistem, bukan di `.venv`.
- **Bug oversubscription CPU di notebook 01 (run pertama)**:
  `RandomizedSearchCV(n_jobs=-1)` dengan estimator yang juga diset
  `n_jobs=-1` menyebabkan setiap worker proses mencoba merebut semua 8
  core sekaligus. Diperbaiki dengan `n_jobs=1` pada estimator di dalam
  search.
- **Ruang pencarian awal kebesaran**: boosting multiclass XGBoost/LightGBM
  membangun `n_estimators x n_classes` pohon per fit (35 kelas) — jauh
  lebih mahal dari RandomForest. `N_ITER` diturunkan 25→12, `cv` 3→2,
  batas atas `n_estimators`/`max_depth` dipersempit. **Celah metodologis
  ini masih belum ditutup** — kesimpulan "tuning tidak membantu
  signifikan" di notebook 01 belum teruji dengan pencarian penuh.
- **Kernel Jupyter menumpuk jika tidak dibersihkan**: `jupyter nbconvert
  --execute` yang dijalankan berkali-kali lewat background task
  meninggalkan proses `ipykernel_launcher`/`resource_tracker` yang
  TIDAK otomatis berhenti setelah selesai. Di sesi ini sempat menumpuk
  jadi **19 proses orphan** yang rebutan CPU dan bikin notebook
  berikutnya nyaris tidak maju (timeout 1800s tanpa progres nyata,
  padahal notebook yang sama sebelumnya selesai dalam hitungan menit).
  Solusi: `Stop-Process` semua proses `python.exe` yang command line-nya
  mengandung `ipykernel_launcher`/`resource_tracker` SEBELUM menjalankan
  eksekusi notebook berikutnya — lakukan ini secara rutin kalau
  menjalankan banyak notebook berturut-turut di sesi yang sama.
- **Desain two-stage awalnya cuma binary** (PASS vs non-PASS) — atas
  masukan lanjutan, ditambahkan varian kaskade kategori (Pass →
  Partscore/Game/SmallSlam/GrandSlam → kontrak persis, memakai
  `target_category` yang sudah ada di dataset).
- **Kebocoran split lewat pasangan open/closed-room** — lihat bagian
  "⚠️ Perbaikan kebocoran split" di atas. Ini temuan paling signifikan
  dari sesi ini; mengubah beberapa kesimpulan eksperimen (lihat
  "Ringkasan hasil").

## Ringkasan hasil

Semua angka val set. **Kolom "Sebelum" = split lama (bocor), "Sesudah" =
split group-aware (diperbaiki, 2026-07-10) — keduanya ditampilkan supaya
terlihat jelas kesimpulan mana yang berubah.**

Baseline XGBoost single-stage/no-resampling di val set: **51.9% → 51.3%**
accuracy, F1 macro **0.306 → 0.269**, F1 weighted **0.491 → 0.469**.

| # | Kandidat terbaik | Accuracy (sblm→ssdh) | F1 macro (sblm→ssdh) | F1 weighted (sblm→ssdh) | Status kesimpulan |
|---|---|---|---|---|---|
| 01 | LightGBM tuned → **XGBoost tuned** | 52.6%→51.1% (LGBM) / 52.1% (XGB) | 0.308→0.232 (LGBM) / 0.274 (XGB) | 0.494→0.467 (LGBM) / **0.481** (XGB) | **BERUBAH** — LightGBM tuning ternyata memburuk di F1 macro setelah split diperbaiki; parameter hasil tuning lama tidak valid dipakai |
| 02 | XGBoost+RandomOverSampler → **LightGBM+class_weight** | 51.7%→52.1% | 0.335→0.309 | 0.508→0.480 | **BERGESER** — pemenang F1 macro berpindah model; arah temuan (resampling/weighting > baseline) tetap sama |
| 03 | Kaskade kategori (tetap) | 51.8-52.4%→49.1-51.3% | 0.312→0.269 | 0.508→**0.491** | **MENYUSUT BANYAK** — top-3 yang dulu melonjak (77.9% vs 76.3%) sekarang nyaris sama (76.3% vs 75.1%); sebagian besar keunggulan itu ternyata dari kebocoran |
| 04 | Tidak ada → **XGBoost 164+23 fitur** | 51.7%→51.2% | 0.299→**0.280** | 0.488→**0.471** | **BERBALIK ARAH** — fitur baru dulu terlihat merugikan, sekarang menunjukkan perbaikan kecil tapi konsisten |
| 05 | **XGBoost+sample_weight+DART** (tetap) | 50.6%→49.2% | 0.343→0.322 | 0.517→0.498 | **BERTAHAN** — satu-satunya juara yang tidak berubah; margin kemenangannya relatif terhadap runner-up malah sedikit lebih besar |

**Kandidat terkuat keseluruhan tidak berubah: XGBoost + sample_weight +
DART** (notebook 05) — F1 macro 0.322, F1 weighted 0.498, robust terhadap
perbaikan split (satu-satunya dari 5 eksperimen yang kesimpulannya
persis sama sebelum & sesudah). Runner-up: LightGBM + class_weight
(notebook 02) dan kaskade kategori (notebook 03), keduanya F1 weighted
~0.48-0.49.

**Pelajaran metodologis paling penting dari sesi ini**: 3 dari 5
kesimpulan eksperimen (01, 03, 04) berubah — dua di antaranya (03: makin
kecil, 04: berbalik arah) cukup material untuk mengubah rekomendasi
praktis. Ini menegaskan pentingnya split yang bebas kebocoran SEBELUM
membandingkan teknik, bukan sesudahnya — kalau split diperbaiki belakangan
setelah kesimpulan sudah ditulis di skripsi, revisi akan jauh lebih mahal
daripada mengulang 5 notebook di sesi ini.

**Belum dikerjakan**: re-tuning penuh (celah metodologis notebook 01 yang
belum ditutup), fokal loss custom, dan validasi kandidat-kandidat di atas
di **test set** (baru divalidasi di val set — aturan main mewajibkan test
set hanya disentuh sekali untuk kandidat final).
