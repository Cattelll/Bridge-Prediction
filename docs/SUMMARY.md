# Ringkasan Proyek

Penelitian mengklasifikasikan apakah kontrak Bridge yang dibid manusia cocok dengan salah satu kontrak par Double-Dummy Solver. Model yang dibandingkan adalah Random Forest, XGBoost, dan LightGBM.

## Dataset utama

Notebook 03 membangun dua dataset sekaligus dari data LIN dan PBN. Keduanya berbagi board, label `matches_par_contract`, deduplikasi, dan split deal-aware:

- `data/processed/`: 174 fitur, yaitu 164 fitur kanonik dan 10 fitur jumlah trik DDS.
- `data/processed_no_dds/`: 164 fitur kanonik untuk pemodelan utama; fitur trik DDS dan rincian kontrak par tidak diberikan sebagai input model.

DDS v4 tetap digunakan untuk membentuk label berdasarkan level, strain, status double/redouble, dan sisi declarer. Dataset berisi 50.227 baris, 24,56% positif, dengan split 35.161 / 7.534 / 7.532. Deal yang sama tidak melintasi train, validation, dan test.

## Alur notebook

Notebook 01–02 memeriksa parsing dan contoh record mentah. Notebook 03 membentuk label DDS, dua varian dataset, split bersama, dan analisis fitur noDDS. Notebook 04 melatih baseline, Notebook 05 mengevaluasinya, Notebook 06 menguji kandidat, dan Notebook 07 menyajikan evaluasi final.

## Hasil

Skor yang tersimpan saat ini berasal dari pipeline terdahulu dengan 174 fitur. Skor tersebut bukan estimasi untuk pipeline utama 164 fitur. Jalankan Notebook 04–07 secara berurutan untuk menghasilkan metrik baru. Pemilihan model memakai AP validation; test hanya mengestimasi kandidat yang sudah dikunci.
