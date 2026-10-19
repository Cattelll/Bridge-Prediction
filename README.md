# Klasifikasi Kecocokan Kontrak Par pada Contract Bridge

Penelitian ini membandingkan Random Forest, XGBoost, dan LightGBM untuk
mengklasifikasikan apakah kontrak yang dibid cocok dengan salah satu kontrak
par Double-Dummy Solver (DDS).

## Definisi label dan data

`matches_par_contract = 1` bila kontrak aktual cocok dengan salah satu kontrak
par DDS pada level, strain, status double/redouble, dan sisi declarer (NS/EW).
Label 0 berarti tidak cocok. DDS v4 menyimpan semua alternatif kontrak par.

Dataset terbaru berisi **50.227 baris**, 606 file LIN dan 1.390 file PBN, serta
174 fitur model (164 fitur kanonik + 10 fitur trik DD). Positif 24,56%; rasio
negatif:positif sekitar 3,1:1. DDS tetap diperlukan untuk label, tetapi sembilan
kolom par DDS yang langsung menentukan label tidak menjadi fitur.

Split 70/15/15 menggunakan `StratifiedGroupKFold` dengan grup berupa tanda
tangan kanonik seluruh 52 kartu. Salinan deal dari file/sumber berbeda selalu
berada di partisi yang sama. Record dengan fitur dan label identik dideduplikasi;
record dengan fitur ringkas sama tetapi label berbeda tetap disimpan.

## Alur penelitian

LIN (606 file) dan PBN (1.390 file) adalah format sumber yang digabung setelah preprocessing. Penelitian membandingkan dua **kondisi dataset fitur**: dengan 10 fitur jumlah trik DDS (174 fitur) dan tanpa 10 fitur tersebut (164 fitur). Kedua kondisi menggunakan board, label DDS, dan split deal-aware yang identik. Istilah "tanpa DDS" berarti tanpa fitur trik DDS; DDS tetap membentuk target pada kedua kondisi.

Alurnya: **LIN/PBN -> Preprocessing -> Ekstraksi -> DDS -> dua kondisi dataset -> Split -> Random Forest, XGBoost, LightGBM -> Analisis hasil -> Evaluasi final**. Rincian tahap ada di [Tahapan Penelitian](docs/TAHAPAN_PENELITIAN.md).

Model dipilih pada validation menggunakan AP sebagai metrik utama. Setelah kandidat dan threshold dikunci, test digunakan untuk evaluasi final.

## Hasil run terbaru

Model dipilih berdasarkan validation set. Tabel test adalah estimasi akhir untuk
kandidat yang sudah dikunci; peringkat antar kandidat pada test hanya deskriptif.
Metrik AP adalah Average Precision (AP; disebut PR-AUC/AP di tabel).

**Baseline dari `configs/config.yaml`:**

| Model | Accuracy | F1 macro | ROC-AUC | AP (PR-AUC) |
|---|---:|---:|---:|---:|
| Random Forest | 0,8032 | 0,7095 | 0,7812 | 0,5888 |
| XGBoost | 0,8089 | 0,6947 | 0,7921 | 0,5946 |
| LightGBM | 0,7916 | 0,7295 | 0,8153 | 0,6285 |

Dari kandidat yang diuji, **Random Forest Exp E** terpilih berdasarkan AP validation (0,6524). Pada test: AP
0,6383, accuracy 0,8109, F1 macro 0,7309. **XGBoost Exp A** memiliki accuracy
validation tertinggi; pada test accuracy-nya 0,8181. Ringkasan semua kandidat ada di
[`outputs/results/REPORT.md`](outputs/results/REPORT.md).

### Ablasi fitur trik DDS

Eksperimen [`2026-10-16/no_dds_feature_ablation`](experiments/2026-10-16/no_dds_feature_ablation/README.md) memakai baris dan split identik untuk
164 vs 174 fitur. Di test, F1 macro meningkat +10,96 pp (RF), +15,02 pp (XGBoost),
dan +9,27 pp (LightGBM) saat 10 fitur trik DD disertakan. Target tetap dibuat
dengan DDS pada kedua kondisi; ablasi ini mengukur fitur, bukan menghilangkan DDS
dari definisi label.

## Menjalankan pipeline

Gunakan Python 3.12 dan environment proyek. Cache DDS v4 harus tersedia; bila
belum ada, buat sekali dari data mentah.

```powershell
python scripts/build_dds_cache.py
python scripts/run_pipeline.py
```

`run_pipeline.py` membangun dataset LIN+PBN dengan split berbasis deal dan
melatih tiga baseline. Notebook resmi 01 sampai 06 mencakup EDA, tuning, serta analisis
SHAP. Jalankan dari repo root secara berurutan bila ingin mereproduksi analisis
lengkap.

## Struktur

```text
src/parser/       LIN dan PBN ke BoardRecord
src/features/     fitur kanonik dan fitur/label DDS
src/preprocessing/label, deduplikasi, split deal-aware
src/models/       wrapper Random Forest, XGBoost, LightGBM
src/evaluation/   metrik biner
notebooks/        pipeline riset resmi notebook 01 sampai 06
scripts/          build cache, pipeline baseline, validasi parser/fitur
data/             raw dan processed; tidak disimpan di Git
outputs/          model dan hasil pipeline; tidak disimpan di Git
```

## Dokumentasi

- [Tahapan penelitian](docs/TAHAPAN_PENELITIAN.md)
- [Ringkasan proyek](docs/SUMMARY.md)
- [Arsitektur pipeline](docs/ARCHITECTURE.md)
- [Data dictionary fitur](docs/FEATURES.md)
- [Algoritma](docs/ALGORITHMS.md)
- [Eksperimen dan hasil](docs/EXPERIMENTS.md)
- [Riwayat keputusan](CLAUDE.md)
