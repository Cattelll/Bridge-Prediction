# Ablasi Fitur Trik DDS - 2026-10-16

Catatan eksperimen historis; proses aktif saat ini membangun kedua varian dataset dari Notebook 01.

## Rancangan

Membandingkan 164 fitur kanonik dengan 174 fitur (ditambah 10 jumlah trik DDS)
untuk RF, XGBoost, dan LightGBM. Label tetap berasal dari DDS v4 pada kedua
kondisi; eksperimen ini hanya mengukur nilai tambah fitur trik DD. Model memakai
konfigurasi `configs/config.yaml`, tanpa tuning.

Perbandingan memakai observasi dan partisi yang sama: 50.227 board, 35.161 train,
7.534 validation, 7.532 test; prevalensi positif 24,56%. Dataset no-DDS dibuat
dengan memproyeksikan kolom DDS dari dataset resmi setelah split deal-aware
dibentuk, sehingga tidak menghitung split secara terpisah.

## Hasil test

| Model | Fitur | Accuracy | F1 macro | F1 positif | Recall positif | ROC-AUC | AP (PR-AUC) |
|---|---|---:|---:|---:|---:|---:|---:|
| RF | 164, tanpa trik DD | 75,00% | 0,5999 | 0,3549 | 0,2802 | 0,7051 | 0,4356 |
| RF | 174, dengan trik DD | 80,32% | 0,7095 | 0,5446 | 0,4792 | 0,7812 | 0,5888 |
| XGBoost | 164, tanpa trik DD | 76,00% | 0,5445 | 0,2313 | 0,1471 | 0,7086 | 0,4386 |
| XGBoost | 174, dengan trik DD | 80,89% | 0,6947 | 0,5080 | 0,4018 | 0,7921 | 0,5946 |
| LightGBM | 164, tanpa trik DD | 71,40% | 0,6367 | 0,4692 | 0,5149 | 0,7046 | 0,4180 |
| LightGBM | 174, dengan trik DD | 79,16% | 0,7295 | 0,5999 | 0,6366 | 0,8153 | 0,6285 |

Dengan AP sebagai metrik utama, penambahan fitur trik DDS menaikkan AP **15,32 pp** pada RF, **15,60 pp** pada XGBoost, dan **21,05 pp** pada LightGBM. F1 macro juga naik **10,96 pp**, **15,02 pp**, dan **9,27 pp** berturut-turut. AP juga meningkat pada ketiga
model. Selisih di atas berlaku pada satu split dan satu konfigurasi per model;
ini bukan estimasi variasi antar split.

## SHAP

SHAP dihitung pada 100 observasi test yang sama. Fitur trik DDS menyumbang sekitar
24,6% dari jumlah mean absolute SHAP RF, 30,8% XGBoost, dan 32,2% LightGBM pada
kondisi 174 fitur. Enam trik DD masuk top 10 RF dan XGBoost; tujuh masuk top 10
LightGBM. RF memakai aproksimasi Saabas; XGBoost dan LightGBM memakai TreeSHAP
exact. SHAP menunjukkan perilaku model, bukan sebab-akibat.

## Artefak

- `outputs/validation_comparison.csv`, `outputs/test_comparison.csv`
- `outputs/validation_delta_no_dds_minus_dds.csv`, `outputs/test_delta_no_dds_minus_dds.csv`
- `outputs/shap_mean_absolute_importance.csv`, `outputs/shap_top_features.png`
- `outputs/best_model_confusion_matrix.png`, `outputs/summary.json`

Model representatif untuk confusion matrix dipilih berdasarkan AP validation. Notebook 01 membangun dataset dengan DDS dan noDDS bersama-sama, mempertahankan label serta split yang identik. Skrip `scripts/build_datasets.py` juga membangun kedua keluaran melalui jalur yang sama.
Skrip membangun ulang split resmi dari sumber LIN+PBN dan cache DDS v4 sebelum
membuat proyeksi tanpa fitur DDS; model tidak dilatih oleh skrip tersebut.
