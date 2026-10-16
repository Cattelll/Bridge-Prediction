# Ringkasan Proyek

Penelitian membandingkan Random Forest, XGBoost, dan LightGBM untuk memprediksi
apakah kontrak Bridge yang dibid manusia cocok dengan salah satu kontrak par
Double-Dummy Solver.

## Target dan dataset

`matches_par_contract` bernilai 1 bila level, strain, status double/redouble, dan
sisi declarer kontrak aktual cocok dengan salah satu kontrak par DDS v4. Dataset
terbaru berisi 50.227 baris dari 606 file LIN dan 1.390 file PBN, dengan 174
fitur model. Positif 24,56%; rasio negatif:positif sekitar 3,1:1.

Split 70/15/15 (35.161 / 7.534 / 7.532) memakai tanda tangan deal 52 kartu
sebagai grup lintas semua sumber. Hanya record dengan fitur dan label identik
yang dihapus sebagai duplikat.

## Metrik dan hasil terbaru

AP (Average Precision) di validation adalah metrik utama seleksi karena fokusnya
kelas positif; F1 macro, F1 positif, precision/recall, accuracy, dan ROC-AUC
dilengkapi sebagai metrik lain. Test hanya mengestimasi kandidat yang dipilih.

Baseline test: RF accuracy 0,8032 / F1 macro 0,7095 / AP 0,5888; XGBoost
0,8089 / 0,6947 / 0,5946; LightGBM 0,7916 / 0,7295 / 0,6285. Random Forest
Exp E tertinggi pada AP validation (0,6524); pada test AP 0,6383, F1 macro
0,7309, accuracy 0,8109. XGBoost Exp A memiliki accuracy validation tertinggi
(test accuracy 0,8181).

Ablasi 10 fitur trik DD pada split sama menaikkan AP test 15,32?21,05 pp
bergantung model. DDS tetap digunakan untuk membuat label pada semua kondisi.
Rincian lengkap: [Eksperimen](EXPERIMENTS.md), [fitur](FEATURES.md),
[arsitektur](ARCHITECTURE.md).
