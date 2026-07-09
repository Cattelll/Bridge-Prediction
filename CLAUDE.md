# CLAUDE.md — Bridge Contract Prediction

## Ringkasan Proyek
Penelitian skripsi membandingkan tiga algoritma ensemble berbasis pohon
(Random Forest, XGBoost, LightGBM) untuk memprediksi kontrak optimal
Contract Bridge dari rekaman BBO LIN format.

## Status Proyek (Juli 2026)
Parser, feature engineering, training, dan evaluasi sudah diimplementasi.
Dataset mentah diperluas dari 411 → **506 file .lin** (10.223 papan setelah
dedup); `scripts/run_pipeline.py` sudah di-retrain ulang di atas dataset ini
pada 2026-07-09 dan `outputs/models/*.pkl` + `outputs/results/*_test.json`
mencerminkan hasil terbaru. **Belum dikerjakan:** notebook 01–04 belum
dieksekusi ulang di dataset yang diperluas ini — cell output & visualisasi
PNG di dalamnya (termasuk `nb04_summary.json` yang dipakai `scripts/report.py`)
masih dari run lama (6.963 papan/156 fitur). Jalankan ulang notebook 01→04
sebelum mengklaim hasil sepenuhnya konsisten. Ringkasan lengkap di
[docs/SUMMARY.md](docs/SUMMARY.md).

---

## Batas Ruang Lingkup

### Yang ada di dalam ruang lingkup
- Tiga model tree-based: RF, XGBoost, LightGBM
- 164 fitur dari data kartu + lelang
- Target utama: `target_base` (36 kelas: PASS + 35 kontrak)
- Evaluasi: accuracy, F1 macro/weighted, top-k accuracy, SHAP

### Yang di luar ruang lingkup — jangan ditambahkan tanpa diskusi
- Neural network / deep learning
- Double-dummy solver (optimal contract secara matematis)
- Reinforcement learning / game theory
- Real-time / streaming prediction
- Web app / API serving
- Target `target` dengan marker doubled/redoubled (66 kelas) sebagai
  primary — gunakan hanya untuk analisis tambahan

---

## Environment

```
Python  : 3.12
Kernel  : "Python 3.12 (Bridge)"  — didaftarkan via ipykernel
Catatan : .venv diblokir Windows Application Control di mesin ini
          Gunakan Python sistem langsung
```

Install dependencies:
```powershell
pip install -e ".[notebook,dev]"
```

Daftarkan Jupyter kernel:
```powershell
python -m ipykernel install --user --name bridge --display-name "Python 3.12 (Bridge)"
```

---

## Struktur Repository

```
src/
  parser/
    lin_parser.py          LINParser, BoardRecord, Hand; tokeniser BBO LIN
  features/
    engineer.py            164 fitur: extract_features(board) → dict
  preprocessing/
    dataset_builder.py     build_dataset(), load_splits()
  models/
    base.py                BaseModel ABC
    random_forest.py       RFModel (sklearn, class_weight=balanced)
    xgboost_model.py       XGBModel
    lightgbm_model.py      LGBMModel
  evaluation/
    metrics.py             evaluate(), compare_models(), save_results()

notebooks/
  01_data_extraction.ipynb    Data extraction: parsing LIN → dataset CSV (data/processed/)
  02_eda_features.ipynb       EDA & feature engineering (164 fitur)
  03_modeling.ipynb           Training + learning curve
  04_evaluation.ipynb         Evaluasi final, SHAP, radar

scripts/
  run_pipeline.py             Pipeline lengkap: parse → train → eval
  report.py                   Generate REPORT.md dari JSON hasil tersimpan
  validate_parser.py          Smoke test parser LIN
  validate_features.py        Validasi output feature engineering

docs/
  SUMMARY.md                   Ringkasan proyek satu halaman
  ARCHITECTURE.md              Alur pipeline, tanggung jawab modul, keputusan desain
  FEATURES.md                  Data dictionary lengkap 164 fitur

configs/config.yaml           Hyperparameter + path (sumber kebenaran)
data/raw/                     506 file .lin (~10.223 board setelah dedup) — tidak di-git
data/processed/               CSV split + artefak encoder — tidak di-git
outputs/models/               Model tersimpan .pkl — tidak di-git
outputs/results/               PNG visualisasi + JSON hasil — tidak di-git
```

---

## Pipeline Data (urutan)

```
1. LINParser.parse_directory("data/raw/")  →  list[BoardRecord]
2. extract_features(board)                 →  dict (164 fitur + metadata + 3 target)
3. build_dataset()                         →  deduplicate, encode label, split 70/15/15
4. Model.fit(X_train, y_train)
5. evaluate(y_true, y_pred, y_proba, le)   →  dict metrik
```

Semua parameter (path, hyperparameter, seed) diambil dari `configs/config.yaml`.

---

## Kelompok Fitur (164 total)

| Kelompok | Prefix | Jml | Keterangan |
|----------|--------|-----|------------|
| Per-seat hand | `N_`, `E_`, `S_`, `W_` | 96 | HCP total, HCP per suit, panjang suit, stopper, controls, LTC, balanced, void/singleton/doubleton, longest suit |
| Partnership | `ns_`, `ew_` | 44 | HCP gabungan, LTC, fit per suit, has_fit (≥8), best suit, NT stoppers, both_balanced |
| HCP advantage | `hcp_ns_advantage` | 1 | `ns_hcp − ew_hcp` |
| Deal context | `dealer_`, `vuln_` | 8 | dealer one-hot ×4, vulnerability one-hot ×4 |
| Auction | `auction_`, `opening_` | 15 | panjang lelang, competitive, ns/ew bid count, doubled, opening level/strain, alerts |

Metadata (prefix `_`) tidak digunakan sebagai fitur ML:
`_board_number`, `_source_file`, `_room`, `_declarer`, `_result`, `_tricks_made`

---

## Target Variable

| Kolom | Kelas | Keterangan |
|-------|-------|------------|
| `target_base` | 36 | **Primary.** "PASS" atau "{level}{strain}" misal "3N", "4S" |
| `target` | ≤66 | Termasuk "x"/"xx" untuk doubled/redoubled |
| `target_category` | 5 | Pass / Partscore / Game / SmallSlam / GrandSlam |

---

## Hasil Final

*(test set, retrain 2026-07-09 di atas 506 file / 10.223 papan / 164 fitur / 35 kelas,
`n_estimators=300` untuk ketiga model — disamakan dari RF sebelumnya 200)*

| Model | Accuracy | Top-3 | Top-5 | F1 Macro | F1 Weighted |
|-------|----------|-------|-------|----------|-------------|
| RandomForest | 46.3% | 75.7% | 85.8% | 0.244 | 0.448 |
| **XGBoost** | **52.9%** | **78.4%** | **86.6%** | 0.275 | **0.500** |
| LightGBM | 51.7% | 76.9% | 85.3% | **0.280** | 0.486 |

XGBoost unggul accuracy, top-k, & F1 weighted; LightGBM unggul F1 Macro (class imbalance).
Sumber: `outputs/results/test_comparison.csv`. Menaikkan RF dari 200→300 trees tidak
memberi perbaikan berarti (accuracy/F1 turun ~0.1pp, top-k naik ~0.5-1pp — level noise);
disamakan demi konsistensi eksperimen, bukan karena ada gain akurasi.

---

## Konvensi Kode

- Semua fitur yang masuk ke model harus numerik/biner — tidak ada kolom string
- Daftar kolom fitur tersimpan di `data/processed/feature_columns.json`;
  gunakan `load_splits()` agar konsisten
- Hyperparameter ada di `configs/config.yaml` — tidak boleh hardcode di notebook
- Model disimpan ke `outputs/models/` dalam format `.pkl`
- Hasil evaluasi disimpan ke `outputs/results/` dalam format `.json` dan `.png`

---

## Invariant — Jangan Ubah Tanpa Alasan Kuat

| Apa | Alasan |
|-----|--------|
| Token handling di `lin_parser.py` | Divalidasi pada 10.223 board (506 file) tanpa error parsing |
| Random seed `42` | Hasil dalam notebook bergantung pada split ini |
| Urutan kolom fitur | Model .pkl terserialisasi dengan urutan ini |
| `target_base` sebagai target utama | Semua model dilatih dan dievaluasi di atas ini |
| Rasio split 70/15/15 | Digunakan di semua perbandingan model |
