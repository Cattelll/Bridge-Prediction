# Percobaan 2026-08-31

Lanjutan dari sesi audit + pembersihan `notebooks/01–06` (commit `9a7c9ae`,
`0325d40`). Pipeline resmi memakai **semua 182 fitur** apa adanya — tidak ada
seleksi fitur algoritmik; SHAP di `notebooks/04` & `06` cuma untuk
interpretasi. Sesi ini menguji apakah seleksi fitur bisa memberi model yang
lebih ringkas tanpa kehilangan performa.

## Daftar percobaan

| # | Notebook | Tujuan | Status |
|---|----------|--------|--------|
| 1 | [01_shap_feature_selection.ipynb](01_shap_feature_selection.ipynb) | Ranking fitur via `mean \|SHAP value\|` (LightGBM & XGBoost, sub-sampel train 2.000 baris) → retrain di top-k (k ∈ {15…182}) → bandingkan ke baseline 182 fitur di val set; analisis dampak per-kelas untuk kontrak langka; evaluasi test set sekali untuk kandidat terpilih | **done** |

## Aturan main

- Val set untuk iterasi & memilih `k`; test set disentuh **sekali** di akhir.
- Prioritas metrik: **F1 macro** (class imbalance ekstrem), lalu accuracy.
- Tidak mengubah `data/processed/` atau `outputs/models/` resmi — output ke
  `experiments/2026-08-31/outputs/`.
- Hyperparameter dari `configs/config.yaml` (LightGBM `class_weight="balanced"`,
  XGBoost default) — sama seperti `notebooks/03`.

## Hasil (01 — seleksi fitur SHAP)

**Test set (disentuh sekali):**

| Model | k | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|---|
| LightGBM (baseline) | 182 | 56.4% | 0.4101 | 0.557 |
| **LightGBM top-40 SHAP** | **40** | 56.1% | 0.4142 | 0.560 |
| XGBoost (baseline) | 182 | 57.5% | 0.3901 | 0.554 |
| XGBoost top-25 SHAP | 25 | 57.7% | 0.3917 | 0.557 |

**Temuan:**
1. Top-40 fitur SHAP (LightGBM) **setara** baseline 182 fitur — selisih F1
   macro +0.4pp berada di dalam noise urutan kolom (~±0.8pp). "Setara dengan
   40 fitur (78% lebih sedikit)" tetap berharga: model lebih ringkas.
2. Kurva F1 macro vs k **datar mulai k ≈ 40** (LightGBM), k ≈ 25 (XGBoost).
   Fitur di atas ~40 tidak menambah apa-apa; XGBoost malah sedikit turun.
3. **Redundansi per-seat besar**: top-40 menyimpan hanya 5/96 fitur per-kursi,
   0/8 dealer/vuln, tapi 12/15 auction & 13/18 DDS. Agregat partnership
   menangkap hampir semua sinyal per-kursi.
4. **Sirkularitas auction terkonfirmasi**: 3 fitur SHAP teratas =
   `auction_ew_bids`, `auction_len`, `auction_ns_bids` (mean |SHAP| ~2× fitur
   berikutnya). Keduanya (LightGBM & XGBoost) sepakat.
5. Kontrak langka **tidak rusak** — net F1 macro sedikit naik di top-40.

**Rekomendasi**: JANGAN promosikan ke `notebooks/` (gain di dalam noise,
ongkos mengubah invariant tidak sepadan). Layak masuk skripsi sebagai analisis
fitur. Tindak lanjut: **ablasi eksplisit fitur auction sirkular**.

Detail: `outputs/shap_feature_selection/{summary.json, test_comparison.csv,
sweep_val.csv, per_class_val.csv, shap_feature_ranking.csv}` + 3 PNG.
