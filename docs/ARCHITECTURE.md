# Arsitektur & Alur Pipeline

Dokumen ini menjelaskan bagaimana data mengalir dari file `.lin` mentah
sampai menjadi model terlatih dan hasil evaluasi. Untuk ringkasan
proyek lihat [README.md](../README.md); untuk daftar lengkap fitur lihat
[FEATURES.md](FEATURES.md).

## Diagram Alur

```
data/raw/*.lin
      │
      ▼
┌─────────────────────┐
│ src/parser/          │  LINParser.parse_directory()
│   lin_parser.py       │  → tokenise LIN → BoardRecord per papan
└─────────────────────┘
      │  list[BoardRecord]
      ▼
┌─────────────────────┐
│ src/features/         │  extract_features(board)
│   engineer.py         │  → dict 164 fitur + metadata + 3 target
└─────────────────────┘
      │  list[dict]
      ▼
┌─────────────────────┐
│ src/preprocessing/    │  build_dataset()
│   dataset_builder.py  │  → dedup, drop kelas <2 sampel, label encode,
│                       │    split stratified 70/15/15
└─────────────────────┘
      │  train.csv / val.csv / test.csv
      ▼                    (data/processed/)
┌─────────────────────┐
│ src/models/           │  RFModel / XGBModel / LGBMModel (BaseModel)
│   *.py                │  .fit() → .pkl (outputs/models/)
└─────────────────────┘
      │  y_pred, y_proba
      ▼
┌─────────────────────┐
│ src/evaluation/       │  evaluate() → accuracy, F1, top-k, confusion
│   metrics.py          │  matrix, SHAP → outputs/results/*.json, *.png
└─────────────────────┘
```

Orkestrasi urutan di atas ada di `scripts/run_pipeline.py` (jalur non-interaktif)
dan di notebook 01→04 (jalur eksploratif/analisis). Keduanya memanggil fungsi
yang sama di `src/`, tidak ada logika duplikat.

## Tanggung Jawab Modul

| Modul | Tanggung jawab | Tidak melakukan |
|-------|----------------|------------------|
| `src/parser/lin_parser.py` | Tokenise teks LIN, rekonstruksi 4 tangan (termasuk infer tangan ke-4 yang hilang), derive kontrak dari lelang (level, strain, declarer, doubled/redoubled) | Tidak menghitung fitur ML apa pun |
| `src/features/engineer.py` | Hitung 164 fitur numerik/biner dari `BoardRecord`, plus 3 kolom target | Tidak melakukan parsing atau I/O file |
| `src/preprocessing/dataset_builder.py` | Gabungkan parsing+ekstraksi banyak file, bersihkan duplikat & kelas langka, encode label, split stratified, simpan artefak (`feature_columns.json`, `label_encoder.pkl`) | Tidak melatih model |
| `src/models/*.py` | Wrapper tipis (`BaseModel`) di atas sklearn/XGBoost/LightGBM: `fit`, `predict`, `predict_proba`, `save`/`load` | Tidak melakukan feature engineering atau tuning hyperparameter otomatis |
| `src/evaluation/metrics.py` | Hitung metrik (accuracy, precision/recall/F1 macro & weighted, top-k), confusion matrix, simpan JSON | Tidak melatih atau memuat model |

## Keputusan Desain Penting

- **Tree-based ensembles saja (RF/XGBoost/LightGBM).** Cocok untuk fitur
  tabular heterogen (HCP, panjang suit, one-hot) tanpa perlu scaling/embedding —
  lihat batasan "di luar ruang lingkup" di [CLAUDE.md](../CLAUDE.md) untuk
  alasan neural network/RL sengaja tidak dipakai.
- **`target_base` (36 kelas) sebagai target utama**, bukan `target` (66 kelas
  dengan marker `x`/`xx`). Marker doubled/redoubled jarang muncul dan akan
  memperparah class imbalance yang sudah berat; `target` disimpan untuk
  analisis tambahan saja.
- **Kelas dengan <2 sampel dibuang** sebelum split (`dataset_builder.py`)
  agar stratified split & label kontigu (dibutuhkan XGBoost) tetap valid.
- **Seed 42 di semua tahap acak** (split data, `random_state` tiap model) —
  jangan diubah, karena hasil di notebook 03/04 dan tabel di README/CLAUDE.md
  bergantung pada split yang sama persis.
- **Urutan kolom fitur tetap** (`feature_columns.json`) — model `.pkl` yang
  sudah tersimpan mengasumsikan urutan kolom input yang sama.
- **Tangan ke-4 (biasanya East) di-derive dari 52 kartu dikurangi 3 tangan
  yang diketahui** (`_derive_missing_hand`) karena beberapa file LIN BBO tidak
  menuliskan tangan East secara eksplisit.

## BaseModel Interface

Semua model (`RFModel`, `XGBModel`, `LGBMModel`) mengimplementasikan
`src/models/base.py::BaseModel`:

```python
fit(X_train, y_train) -> None
predict(X) -> np.ndarray
predict_proba(X) -> np.ndarray
feature_importances() -> np.ndarray | None
save(path) -> None
load(path) -> BaseModel        # classmethod
```

Kontrak yang sama ini memungkinkan `scripts/run_pipeline.py`, notebook, dan
`src/evaluation/metrics.py::compare_models()` memperlakukan ketiga algoritma
secara seragam.
