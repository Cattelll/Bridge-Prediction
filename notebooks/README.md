# Pipeline Notebook Penelitian

Notebook 01 memeriksa preprocessing dan parsing sumber LIN/PBN. Notebook 02 mengeksplorasi contoh record mentah. **Notebook 03 menjalankan tahap DDS lalu split data**: DDS v4 membentuk label, builder membuat satu assignment split deal-aware, lalu menulis dua proyeksi dataset dari board dan split yang sama:

- `data/processed/`: 174 fitur (164 fitur kanonik + 10 nilai trik DDS).
- `data/processed_no_dds/`: 164 fitur kanonik untuk pemodelan utama.

Dataset noDDS tidak memberikan nilai trik atau rincian par DDS sebagai input model, tetapi tetap mempertahankan label yang dibuat dengan DDS. Hanya noDDS yang diteruskan ke tiga model utama. Notebook 03 memeriksa kesetaraan split dan profil fitur noDDS; notebook 05 dan 06 menangani analisis hasil model.

| Urutan | Notebook | Tahap penelitian |
|---|---|---|
| 01 | 01_data_extraction.ipynb | Preprocessing dan parsing LIN/PBN; melihat contoh board hasil normalisasi. |
| 02 | 02_data_exploration.ipynb | Eksplorasi sampel record mentah LIN/PBN sebelum feature engineering. |
| 03 | 03_dds_dataset_split.ipynb | DDS untuk label, split deal-aware, pembuatan dua proyeksi dataset, dan profil data noDDS. |
| 04 | 04_modeling.ipynb | Melatih baseline Random Forest, XGBoost, dan LightGBM pada noDDS. |
| 05 | 05_evaluation.ipynb | Analisis baseline di validation: metrik, confusion matrix, feature importance, dan SHAP. |
| 06 | 06_improvement_experiments.ipynb | Eksperimen model dan threshold; seleksi hanya berdasarkan validation. |
| 07 | 07_final_evaluation.ipynb | Mengunci kandidat dari validation dan menjalankan evaluasi test final satu kali. |

Jalankan notebook berurutan dari repo root. Notebook 03 membaca cache DDS v4 dan menulis kedua dataset secara berpasangan. Untuk membangunnya dari command line, jalankan `python scripts/build_datasets.py`; skrip ini juga menghasilkan kedua varian. Notebook 04–07 menggunakan `data/processed_no_dds/` untuk modeling dan evaluasi.
