# Urutan Notebook Penelitian

Pipeline utama memakai dataset tanpa fitur jumlah trik DDS untuk pemodelan.
DDS tetap digunakan untuk membentuk label matches_par_contract. Sepuluh fitur
trik DDS tidak dimasukkan ke fitur model.

| Urutan | Notebook | Tahap penelitian |
|---|---|---|
| 01 | 01_data_extraction.ipynb | Parsing LIN/PBN, preprocessing, ekstraksi, DDS untuk label, dan split deal-aware. |
| 02 | 04_dds_feature_ablation.ipynb | Menyiapkan dan memeriksa proyeksi 164 fitur tanpa fitur trik DDS; tidak melatih model. |
| 03 | 02_data_exploration.ipynb | Pemeriksaan tabel board dan dataset noDDS. |
| 04 | 03_feature_analysis.ipynb | Pemeriksaan distribusi 164 fitur noDDS sebelum pemodelan. |
| 05 | 05_modeling.ipynb | Melatih baseline Random Forest, XGBoost, dan LightGBM pada dataset tanpa fitur DDS. |
| 06 | 06_evaluation.ipynb | Evaluasi baseline, confusion matrix, feature importance, dan SHAP. |
| 07 | 07_improvement_experiments.ipynb | Retuning kandidat dan threshold berdasarkan validation. |
| 08 | 08_final_evaluation.ipynb | Memilih kandidat berdasarkan validation dan melaporkan evaluasi test. |

Jalankan notebook sesuai urutan pipeline pada kolom pertama dari repo root.
Notebook 04 (file bernama 04_dds_feature_ablation.ipynb) menyiapkan
data/processed_no_dds/ dari hasil Notebook 01 jika proyeksi belum tersedia.
Notebook 02–03 mengeksplorasi dataset tersebut, lalu Notebook 05–08 memakai
partisi dan fitur yang sama.
