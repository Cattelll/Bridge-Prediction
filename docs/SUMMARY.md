# Ringkasan Proyek

Penelitian mengklasifikasikan apakah kontrak Bridge yang dibid manusia cocok
dengan salah satu kontrak par Double-Dummy Solver. Model yang digunakan adalah
Random Forest, XGBoost, dan LightGBM.

## Dataset utama

Sumber LIN dan PBN digabung setelah preprocessing. DDS v4 menghitung label
matches_par_contract dari level, strain, status double/redouble, dan sisi
declarer. Ekstraksi menghasilkan 164 fitur kanonik ditambah 10 fitur trik DDS;
pipeline utama mengeluarkan 10 fitur trik tersebut, sehingga model dilatih
menggunakan 164 fitur. Label tetap berasal dari DDS.

Dataset memiliki 50.227 baris, 24,56% positif, dan split deal-aware
35.161 / 7.534 / 7.532. Deal yang sama tidak melintasi train, validation,
dan test.

## Alur notebook

Notebook 01 mengekstrak dan membentuk split. Notebook 02–03 memeriksa data dan
fitur. Notebook 04 menyiapkan dataset tanpa fitur trik DDS dan memeriksa
kesamaan label/partisi; pelatihan model tidak dilakukan di sana. Notebook 05
melatih tiga baseline pada 164 fitur. Notebook 06 mengevaluasi baseline,
Notebook 07 menguji kandidat, dan Notebook 08 melaporkan evaluasi final.

## Hasil

Skor yang saat ini tercatat di laporan berasal dari pipeline terdahulu dengan
174 fitur. Skor tersebut bukan estimasi untuk pipeline utama 164 fitur.
Jalankan Notebook 05–08 secara berurutan untuk menghasilkan metrik baru.
Pemilihan model memakai AP validation; test hanya mengestimasi kandidat yang
sudah dikunci.
