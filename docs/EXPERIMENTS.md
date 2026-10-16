# Eksperimen dan Evaluasi

## Status data dan label

Run terbaru memakai label `matches_par_contract` dari DDS v4. Nilai 1 berarti
kontrak aktual cocok dengan salah satu kontrak par pada level, strain, status
double/redouble, dan sisi declarer. Dataset berisi 50.227 observasi, 174 fitur,
dan prevalensi positif 24,56% (negatif:positif sekitar 3,1:1). Split 35.161 /
7.534 / 7.532 memakai tanda tangan deal 52 kartu sebagai grup.

Kolom par DDS yang langsung membentuk label dikeluarkan dari fitur. Sepuluh
fitur jumlah trik DDS diuji melalui ablation pada baris dan split identik.

## Metrik dan aturan seleksi

**Average Precision (AP) pada validation adalah metrik utama pemilihan model**,
karena fokus penelitian adalah mengenali kelas positif. AP dihitung dengan
`average_precision_score` scikit-learn dan ditampilkan sebagai AP (PR-AUC/AP).
F1 macro, F1 positif, precision/recall pada threshold, accuracy, dan ROC-AUC
adalah metrik pelengkap. Threshold keputusan ditentukan di validation. Test
hanya mengestimasi kandidat yang telah dikunci; peringkat semua kandidat pada
test bersifat deskriptif.

### Baseline resmi pada test

| Model | Accuracy | F1 macro | F1 positif | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0,8032 | 0,7095 | 0,5446 | 0,7812 | 0,5888 |
| XGBoost | 0,8089 | 0,6947 | 0,5080 | 0,7921 | 0,5946 |
| LightGBM | 0,7916 | 0,7295 | 0,5999 | 0,8153 | 0,6285 |

Baseline mayoritas sekitar 75,5% accuracy; jangan jadikan accuracy saja sebagai
ukuran performa.

### Kandidat terpilih berdasarkan validation

- **Pilihan utama, AP validation tertinggi:** Random Forest Exp E. AP validation
  0,6524; pada test AP 0,6383, F1 macro 0,7309, accuracy 0,8109.
- **Accuracy validation tertinggi (sekunder):** XGBoost Exp A. Test accuracy
  0,8181; F1 macro 0,7132; AP 0,6256.
- **F1 macro validation tertinggi (sekunder):** LightGBM Exp C. Test F1 macro
  0,7361; AP 0,6326.

Tabel delapan kandidat test ada di
[`outputs/results/nb06_final_comparison.csv`](../outputs/results/nb06_final_comparison.csv)
dan bersifat deskriptif. Hasil sebelum split global deal-aware pada 2026-10-16
tidak dianggap sebagai estimasi final.

## Ablasi 10 fitur trik DDS

Notebook [`2026-10-16/no_dds_feature_ablation`](../experiments/2026-10-16/no_dds_feature_ablation/README.md) mempertahankan label DDS untuk kedua kondisi, tetapi membandingkan 164 fitur
kanonik dengan 174 fitur yang menambah trik DD. Pada test, AP naik **15,32 pp**
(RF), **15,60 pp** (XGBoost), dan **21,05 pp** (LightGBM). F1 macro naik 10,96,
15,02, dan 9,27 pp berturut-turut.

SHAP menjelaskan perilaku model, bukan sebab-akibat. RF memakai aproksimasi
Saabas pada sampel terbatas; XGBoost dan LightGBM memakai TreeSHAP exact.

## Arsip eksperimen lama

Notebook `experiments/2026-07-*` dan `experiments/2026-08-*` berasal dari target
36 kelas atau split sebelum pengelompokan global deal. Gunakan sebagai riwayat,
bukan angka pembanding final pipeline biner sekarang.
