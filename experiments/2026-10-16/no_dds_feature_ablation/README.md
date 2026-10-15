# Ablasi fitur DDS ? 2026-10-16

Notebook: [`01_no_dds_feature_ablation.ipynb`](01_no_dds_feature_ablation.ipynb)

## Tujuan dan rancangan

Menguji kontribusi 10 fitur trik DD (`ns_dd_*`, `ew_dd_*`) pada RandomForest, XGBoost, dan LightGBM. Kondisi dibandingkan pada **baris, label, dan split yang sama**: 164 fitur kanonik tanpa trik DD versus 174 fitur dengan trik DD. Hyperparameter dibaca dari `configs/config.yaml`; tidak ada tuning. Label `matches_par_contract` tetap mengikuti DDS v4: kontrak aktual cocok dengan salah satu kontrak par pada level, strain, status double/redouble, dan sisi declarer. Jadi eksperimen ini mengisolasi nilai tambah fitur trik DD, bukan sistem yang sama sekali tidak memakai DDS.

Notebook memakai data 49.755 board (34.829 train, 7.462 validation, 7.464 test; prevalensi positif 24,3%). Test dipakai setelah konfigurasi model ditetapkan. Hasil dan grafik tersimpan di `outputs/`.

## Hasil test

| Model | Fitur | Accuracy | F1 macro | F1 positif | Recall positif | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| RandomForest | Tanpa DDS (164) | 80,21% | 0,7162 | 0,5600 | 0,5182 | 0,6096 |
| RandomForest | Dengan DDS (174) | 82,34% | 0,7542 | 0,6239 | 0,6025 | 0,6667 |
| XGBoost | Tanpa DDS (164) | 79,41% | 0,6219 | 0,3667 | 0,2453 | 0,5669 |
| XGBoost | Dengan DDS (174) | 83,03% | 0,7287 | 0,5627 | 0,4493 | 0,6774 |
| LightGBM | Tanpa DDS (164) | 76,43% | 0,6994 | 0,5597 | 0,6163 | 0,6001 |
| LightGBM | Dengan DDS (174) | 81,46% | 0,7609 | 0,6477 | 0,7012 | 0,7018 |

Penambahan 10 fitur DDS menaikkan F1 macro pada ketiga model: **+3,81 poin persentase** untuk RF, **+10,68 pp** untuk XGBoost, dan **+6,15 pp** untuk LightGBM. Accuracy juga naik 2,13?5,03 pp. Tanpa DDS, RF menjadi model terbaik di antara ketiganya pada test (F1 macro 0,7162); dengan DDS, LightGBM tetap terbaik pada F1 macro (0,7609), sementara XGBoost memiliki accuracy tertinggi (83,03%). XGBoost tanpa DDS paling terdampak pada kelas positif: recall turun menjadi 24,53%.

## SHAP

SHAP dihitung pada 100 sampel test yang sama per model dan kondisi. Lima dari 10 fitur trik DD masuk top 10 pada RF dan LightGBM; enam masuk top 10 pada XGBoost. Pada model dengan DDS, fitur trik DD menyumbang sekitar 23,6% (RF), 30,5% (LightGBM), dan 30,3% (XGBoost) dari jumlah mean absolute SHAP semua fitur. Angka ini adalah proporsi importance SHAP global, bukan kontribusi terhadap skor akurasi.

Tanpa DDS, `ns_hcp` menjadi fitur teratas di ketiga algoritma. Dengan DDS, fitur jumlah trik seperti `ns_dd_N`, `ns_dd_S`, dan `ew_dd_N` muncul di antara fitur paling berpengaruh. Ini konsisten dengan penurunan metrik ketika fitur trik DD dihilangkan.

**Catatan interpretasi:** SHAP exact dipakai untuk XGBoost dan LightGBM. Exact Tree SHAP untuk RF melampaui batas waktu praktis, jadi RF memakai aproksimasi Saabas. Besaran SHAP lintas algoritma tidak dibandingkan langsung; gunakan peringkat fitur dan perbandingan dalam algoritma yang sama. SHAP menjelaskan perilaku prediksi, bukan sebab-akibat. Hasil berasal dari satu split dan satu konfigurasi per model.

## Artefak

- `outputs/validation_comparison.csv`, `outputs/test_comparison.csv`
- `outputs/validation_delta_no_dds_minus_dds.csv`, `outputs/test_delta_no_dds_minus_dds.csv`
- `outputs/shap_mean_absolute_importance.csv`, `outputs/shap_top_features.png`
- `outputs/best_model_confusion_matrix.png`, `outputs/summary.json`

Dataset resmi `data/processed/` dan model resmi tidak diubah. Untuk membangun ulang dataset no-DDS, jalankan `scripts/build_dataset_no_dds.py`.
