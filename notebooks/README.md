# Urutan Notebook Penelitian

Urutan berikut mengikuti diagram penelitian: data dan DDS, eksplorasi, perbandingan dataset dengan/tanpa fitur DDS, pemodelan, analisis hasil, lalu evaluasi final. LIN dan PBN digabung sebagai sumber data; varian tanpa fitur trik DDS tetap memakai label DDS dan split yang sama.

| Urutan | Notebook | Tahap penelitian |
|---|---|---|
| 01 | `01_data_extraction.ipynb` | Parsing LIN/PBN, preprocessing, ekstraksi, DDS, label, dan split resmi dengan 174 fitur. |
| 02 | `02_data_exploration.ipynb` | Pemeriksaan tabel board dan dataset setelah ekstraksi. |
| 03 | `03_feature_analysis.ipynb` | Pemeriksaan distribusi dan fitur sebelum pemodelan. |
| 04 | `04_dds_feature_ablation.ipynb` | Membandingkan 174 fitur dengan 164 fitur tanpa sepuluh fitur jumlah trik DDS; label dan partisi tetap identik. |
| 05 | `05_modeling.ipynb` | Melatih baseline Random Forest, XGBoost, dan LightGBM pada dataset resmi. |
| 06 | `06_evaluation.ipynb` | Analisis baseline, metrik, feature importance, dan SHAP. |
| 07 | `07_improvement_experiments.ipynb` | Analisis kandidat pada validation. |
| 08 | `08_final_evaluation.ipynb` | Memilih kandidat berdasarkan validation dan melaporkan evaluasi test. |

Jalankan dari repo root. Notebook 04 membangun dataset proyeksi tanpa fitur DDS otomatis bila belum tersedia. Untuk membuatnya secara terpisah, jalankan `python scripts/build_dataset_no_dds.py`.
