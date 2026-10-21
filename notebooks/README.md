# Pipeline Notebook Penelitian

Notebook 01 memeriksa preprocessing dan parsing sampel LIN/PBN. Notebook 02 mengeksplorasi contoh record mentah. **Notebook 03 menjalankan tahap DDS lalu split data**: DDS v4 membentuk label, builder membuat satu assignment split deal-aware, lalu menulis dua proyeksi dataset dari board dan split yang sama:

- `data/processed/`: 174 fitur (164 fitur kanonik + 10 nilai trik DDS).
- `data/processed_no_dds/`: 164 fitur kanonik untuk pemodelan utama.

Dataset noDDS tidak memberikan nilai trik atau rincian par DDS sebagai input model, tetapi tetap mempertahankan label yang dibuat dengan DDS. Kedua varian diteruskan ke Random Forest, XGBoost, dan LightGBM dengan algoritma, konfigurasi, label, dan split yang sama agar pengaruh 10 fitur trik DDS dapat dibandingkan. Notebook 03 memeriksa kesetaraan split. Notebook 04 melatih keenam baseline berpasangan; Notebook 05 membandingkan metrik keenam model dan mengukur kontribusi fitur DDS dengan SHAP pada model DDS; Notebook 06 memuat eksperimen tambahan noDDS; Notebook 07 membahas improvement berpasangan pada DDS dan noDDS untuk ketiga algoritma; Notebook 08 mengevaluasi baseline dan kandidat terpilih pada test set.

| Urutan | Notebook | Tahap penelitian |
|---|---|---|
| 01 | 01_data_extraction.ipynb | Preprocessing dan parsing LIN/PBN; melihat contoh board hasil normalisasi. |
| 02 | 02_data_exploration.ipynb | Eksplorasi sampel record mentah LIN/PBN sebelum feature engineering. |
| 03 | 03_dds_dataset_split.ipynb | DDS untuk label, split deal-aware, pembuatan dua proyeksi dataset, dan profil data noDDS. |
| 04 | 04_modeling.ipynb | Melatih RF, XGBoost, dan LightGBM pada DDS serta noDDS dengan konfigurasi yang sama. |
| 05 | 05_evaluation.ipynb | Analisis enam baseline di validation: metrik berpasangan dan porsi kontribusi fitur DDS berdasarkan SHAP pada tiap algoritma DDS. |
| 06 | 06_improvement_experiments.ipynb | Eksperimen tambahan noDDS dan threshold; evaluasi validation saja. |
| 07 | 07_paired_improvement_experiments.ipynb | Membandingkan eksperimen improvement DDS dan noDDS untuk RF, XGBoost, dan LightGBM; seleksi berdasarkan validation, lalu evaluasi kandidat terpilih. |
| 08 | 08_final_evaluation.ipynb | Membandingkan enam baseline serta improvement terpilih pada test set yang sama. |

Jalankan notebook berurutan dari repo root. Notebook 03 membaca cache DDS v4 dan menulis kedua dataset secara berpasangan. Untuk membangunnya dari command line, jalankan `python scripts/build_datasets.py`; skrip ini juga menghasilkan kedua varian. Notebook 04, 05, 07, dan 08 menggunakan kedua varian. Notebook 06 menjalankan eksperimen tambahan noDDS. Notebook 07 menjalankan dan membandingkan eksperimen improvement pada DDS/noDDS.
