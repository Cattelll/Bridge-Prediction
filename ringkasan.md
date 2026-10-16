# Ringkasan Proyek ? Klasifikasi Kecocokan Kontrak Par

*Diperbarui 2026-10-16 setelah perbaikan split deal-aware.*

Penelitian membandingkan Random Forest, XGBoost, dan LightGBM untuk menentukan
apakah kontrak aktual cocok dengan salah satu kontrak par DDS. Label
`matches_par_contract` memeriksa level, strain, status double/redouble, dan sisi
declarer.

## Data dan split

- 606 file LIN + 1.390 file PBN; 50.227 baris.
- 174 fitur: 164 kanonik + 10 jumlah trik DDS.
- Kelas positif 24,56%; rasio negatif:positif sekitar 3,1:1.
- Split 35.161 / 7.534 / 7.532 dikelompokkan lewat tanda tangan 52 kartu
  kanonik; salinan deal lintas file/sumber selalu berada dalam partisi yang sama.
- Deduplikasi membuang hanya record dengan fitur dan label yang identik.

## Metrik dan hasil

AP (Average Precision) pada validation menjadi metrik utama seleksi karena fokus
penelitian adalah kelas positif. F1 macro, F1 positif, precision/recall, accuracy,
dan ROC-AUC tetap dilaporkan. Random Forest Exp E tertinggi pada AP validation
(0,6524); test AP 0,6383, F1 macro 0,7309. XGBoost Exp A tertinggi pada accuracy
validation; test accuracy 0,8181.

Ablasi 10 fitur trik DD pada split sama menaikkan AP test 15,32?21,05 pp dan F1
macro 9,27?15,02 pp. Lihat [hasil eksperimen](docs/EXPERIMENTS.md) dan
[laporan](outputs/results/REPORT.md). Hasil sebelum perbaikan split global adalah
historis, bukan estimasi final.
