# Tahapan Penelitian

Penelitian mengklasifikasikan kecocokan kontrak aktual dengan kontrak par
menggunakan Random Forest, XGBoost, dan LightGBM. Dataset yang digunakan
untuk pemodelan tidak memuat fitur jumlah trik DDS. DDS tetap dipakai untuk
membentuk label.

```mermaid
flowchart TD
    A([Mulai]) --> B1[(Sumber LIN)]
    A --> B2[(Sumber PBN)]
    B1 --> C[Preprocessing dan normalisasi]
    B2 --> C
    C --> D[Ekstraksi fitur kanonik]
    D --> E[(DDS untuk label dan jumlah trik)]
    E --> F[Split data deal-aware<br/>assignment sama untuk kedua varian]
    F --> F1[(Dataset DDS: 174 fitur)]
    F --> F2[(Dataset noDDS: 164 fitur)]
    F2 --> H1[Random Forest]
    F2 --> H2[XGBoost]
    F2 --> H3[LightGBM]
    H1 --> I[Analisis hasil]
    H2 --> I
    H3 --> I
    I --> J[Evaluasi final]
    J --> K([Selesai])
```

## 1. Sumber data dan preprocessing

Sumber mentah terdiri dari 606 file LIN (BBO) dan 1.390 file PBN (arsip
turnamen). Parser menormalkan kedua format menjadi BoardRecord dan
menggabungkannya. Setiap deal divalidasi dan mendapat identitas kanonik dari
susunan 52 kartu.

## 2. Ekstraksi data dan DDS

Ekstraksi menghasilkan 164 fitur kanonik dari tangan, auction, dan kondisi
board. DDS v4 menghitung jumlah trik untuk lima strain dan seluruh alternatif
kontrak par. Label matches_par_contract bernilai 1 bila kontrak aktual cocok
dengan salah satu kontrak par dalam level, strain, status double/redouble,
dan sisi declarer; selain itu 0. Kolom par yang langsung membentuk label tidak
dipakai sebagai fitur.

## 3. Dataset untuk pemodelan

Notebook 03 menghitung label dengan DDS, menghapus duplikat yang identik, lalu membuat satu assignment split deal-aware. Dari board dan split yang sama, notebook menyimpan dua proyeksi: dataset DDS dengan 174 fitur (164 kanonik + 10 fitur trik DDS) dan dataset pemodelan noDDS dengan 164 fitur kanonik. Rekaman dengan fitur dan label identik dideduplikasi; rekaman dengan
fitur ringkas sama tetapi label berbeda tetap dipertahankan.

## 4. Split data

Train, validation, dan test dikelompokkan berdasarkan identitas deal 52 kartu
agar deal yang sama tidak melintasi partisi. Assignment yang sama diterapkan
pada kedua proyeksi dataset; model utama memakai noDDS. Dataset tanpa fitur DDS memiliki
35.161 baris train, 7.534 validation, dan 7.532 test.

## 5. Pemodelan dan analisis

Notebook 04 melatih baseline Random Forest, XGBoost, dan LightGBM. Notebook 05
mengevaluasi baseline. Notebook 06 menguji kandidat dan threshold dengan validation sebagai dasar pemilihan. AP menjadi metrik utama; F1 macro, F1
positif, precision, recall, ROC-AUC, dan accuracy menjadi metrik pendamping.
SHAP dan feature importance dipakai untuk interpretasi prediksi, bukan untuk
menyatakan hubungan sebab-akibat.

## 6. Evaluasi final

Notebook 07 memilih kandidat berdasarkan validation dan menggunakan test
untuk estimasi akhir. Peringkat kandidat lain pada test hanya bersifat
deskriptif. Hasil yang saat ini tersimpan berasal dari pipeline lama dengan
174 fitur; Notebook 04–07 perlu dijalankan ulang untuk memperoleh hasil final
pada dataset 164 fitur.

## Notebook yang merealisasikan tahapan

- 01_data_extraction.ipynb: preprocessing dan parsing LIN/PBN.
- 02_data_exploration.ipynb: eksplorasi sampel record mentah.
- 03_dds_dataset_split.ipynb: DDS untuk label, pembentukan dua dataset, split deal-aware, dan analisis fitur noDDS.
- 04_modeling.ipynb: pelatihan tiga baseline pada dataset 164 fitur noDDS.
- 05_evaluation.ipynb: evaluasi baseline.
- 06_improvement_experiments.ipynb: eksperimen kandidat pada validation.
- 07_final_evaluation.ipynb: pemilihan kandidat dan evaluasi test.
