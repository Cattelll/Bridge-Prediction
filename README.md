# Bridge Contract Prediction

Penelitian perbandingan algoritma machine learning (Random Forest, XGBoost, LightGBM) untuk memprediksi kontrak optimal dalam permainan Bridge menggunakan dataset rekaman BBO LIN format.

## Latar Belakang

Contract Bridge adalah permainan kartu yang melibatkan proses bidding (lelang) untuk menentukan kontrak — level dan jenis suit yang akan dimainkan. Memprediksi kontrak optimal secara otomatis merupakan masalah klasifikasi multikelas dengan 35 kelas target.

Dataset berasal dari **506 file .lin** BBO (Bridge Base Online), menghasilkan **10.223 papan** setelah pembersihan & deduplikasi (35 kelas kontrak, 164 fitur).

## Hasil

*(test set, retrain terakhir pada dataset 506 file / 10.223 papan, `n_estimators=300` di ketiga model)*

| Model | Accuracy | Top-3 Acc | Top-5 Acc | F1 Macro | F1 Weighted |
|-------|----------|-----------|-----------|----------|-------------|
| Random Forest | 46.3% | 75.7% | 85.8% | 0.244 | 0.448 |
| **XGBoost** | **52.9%** | **78.4%** | **86.6%** | 0.275 | **0.500** |
| LightGBM | 51.7% | 76.9% | 85.3% | **0.280** | 0.486 |

> XGBoost unggul dalam Accuracy, Top-k, & F1 Weighted; LightGBM unggul dalam F1 Macro (penanganan class imbalance).
> Lihat [docs/SUMMARY.md](docs/SUMMARY.md) untuk ringkasan lengkap & [outputs/results/REPORT.md](outputs/results/REPORT.md) untuk hasil mentah.

## Struktur Proyek

```
Bridge-Prediction/
├── src/
│   ├── parser/          # LIN parser (BBO format)
│   ├── features/        # Feature engineering (164 fitur)
│   ├── preprocessing/   # Dataset builder & split
│   ├── models/          # RF, XGBoost, LightGBM wrappers
│   └── evaluation/      # Metrics, confusion matrix, SHAP
├── notebooks/
│   ├── 01_data_extraction.ipynb    # Data extraction: parsing LIN → dataset CSV
│   ├── 02_eda_features.ipynb       # EDA & feature engineering (164 fitur)
│   ├── 03_modeling.ipynb           # Training & learning curve
│   └── 04_evaluation.ipynb         # Evaluasi final, SHAP, radar
├── scripts/
│   ├── run_pipeline.py             # Pipeline lengkap (parse → train → eval)
│   ├── validate_parser.py          # Validasi parser LIN
│   └── validate_features.py        # Validasi fitur
├── configs/config.yaml             # Konfigurasi hyperparameter
├── data/
│   ├── raw/             # File .lin BBO (tidak di-track git)
│   └── processed/       # CSV split train/val/test (tidak di-track git)
└── outputs/
    ├── models/          # Model tersimpan (.pkl)
    └── results/         # Visualisasi output (.png)
```

## Fitur (164 total)

| Kelompok | Jumlah | Keterangan |
|----------|--------|------------|
| Per-seat (N/E/S/W) | 96 | HCP, panjang suit, stopper, kontrol, LTC, balanced |
| Partnership (NS/EW) | 44 | HCP gabungan, fit, best suit, NT stopper |
| HCP advantage | 1 | Selisih HCP NS vs EW |
| Deal context | 8 | Dealer (one-hot 4), vulnerability (one-hot 4) |
| Auction | 15 | Panjang lelang, kompetitif, opening level/strain, alert |

## Instalasi

```powershell
# Clone repo
git clone <repo-url>
cd Bridge-Prediction

# Install dependencies (gunakan Python 3.12 sistem)
pip install -e ".[notebook,dev]"
# atau jika pyproject.toml belum ada di working tree:
pip install -r requirements.txt

# Daftarkan Jupyter kernel
python -m ipykernel install --user --name bridge --display-name "Python 3.12 (Bridge)"
```

> **Catatan:** `pyproject.toml` saat ini terhapus di working tree lokal (unstaged) —
> gunakan `requirements.txt` sebagai fallback sampai ini diselesaikan.
>
> Virtual environment `.venv` mungkin diblokir oleh Windows Application
> Control. Gunakan Python sistem secara langsung. Pilih kernel
> **"Python 3.12 (Bridge)"** di Jupyter.

## Penggunaan

### Jalankan pipeline lengkap

```bash
python scripts/run_pipeline.py
```

### Jalankan notebook secara berurutan

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_extraction.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_features.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
```

### Validasi parser

```bash
python scripts/validate_parser.py
python scripts/validate_features.py
```

## Format Data (BBO LIN)

File `.lin` adalah format rekaman permainan Bridge dari Bridge Base Online:

```
vg|tournament_name,...|
qx|o1|md|3SJ9HAQ9...,...,...,...|sv|0|
mb|1H|mb|P|mb|2H|mb|P|mb|4H|mb|P|mb|P|mb|P|
pc|S3|pc|S5|pc|S8|pc|SA|mc|10|
```

Token utama: `vg` (info turnamen), `md` (distribusi kartu), `sv` (vulnerability),
`mb` (bid), `pc` (kartu dimainkan), `mc` (jumlah trick).

## Lingkungan

- Python 3.12
- scikit-learn ≥ 1.4, xgboost ≥ 2.0, lightgbm ≥ 4.0
- shap ≥ 0.45, matplotlib ≥ 3.8, pandas ≥ 2.1

## Dokumentasi Tambahan

- [docs/SUMMARY.md](docs/SUMMARY.md) — ringkasan proyek satu halaman (tujuan, metode, hasil, temuan)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — alur pipeline, tanggung jawab tiap modul, keputusan desain
- [docs/FEATURES.md](docs/FEATURES.md) — data dictionary lengkap 164 fitur

## Lisensi

Untuk keperluan penelitian/skripsi. Dataset BBO LIN bersifat publik dari Bridge Base Online.
