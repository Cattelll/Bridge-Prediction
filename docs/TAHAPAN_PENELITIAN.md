# Tahapan Penelitian

Penelitian membandingkan Random Forest, XGBoost, dan LightGBM untuk klasifikasi
biner kecocokan kontrak par Contract Bridge.

## 1. Data dan target

Sumber data adalah rekaman LIN dari BBO dan arsip PBN. DDS v4 menghitung jumlah
trik serta seluruh alternatif kontrak par. Label `matches_par_contract` bernilai
1 bila kontrak aktual cocok pada level, strain, status double/redouble, dan sisi
declarer; selain itu 0. DDS diperlukan untuk label walaupun 10 fitur jumlah trik
diuji terpisah dalam eksperimen ablation. Kolom par DDS yang langsung menentukan
label tidak dipakai sebagai fitur.

Dataset run terbaru: 606 file LIN, 1.390 file PBN, 50.227 observasi dan 174 fitur.
Kelas positif 24,56% (rasio negatif:positif sekitar 3,1:1; ringan hingga sedang).

## 2. Prapemrosesan dan split

Parser membentuk `BoardRecord`, fitur dihitung per deal, lalu rekaman identik pada
fitur dan label dideduplikasi. Deal mendapat identitas kanonik dari seluruh 52
kartu, tidak bergantung nama sumber atau nomor board. `StratifiedGroupKFold`
membagi berdasarkan identitas deal, sehingga tidak ada salinan deal lintas
train/validation/test. Split run terbaru: 35.161 train, 7.534 validation, 7.532
test.

## 3. Pemodelan dan evaluasi

Tiga model dilatih dengan fitur numerik. Validation dipakai memilih model,
hyperparameter, serta threshold. Test hanya dipakai mengestimasi performa pilihan
yang sudah dikunci. Dilaporkan accuracy, F1 macro, F1 positif, precision/recall,
ROC-AUC, serta AP (Average Precision; sering disebut PR-AUC/AP).

Hasil utama terbaru: Random Forest Exp E terpilih dari validation berdasarkan AP (hasil test AP
0,6383, F1 macro 0,7309). XGBoost Exp A memiliki accuracy validation tertinggi
(hasil test accuracy 0,8181). Ablasi fitur menunjukkan
kenaikan F1 macro saat 10 fitur trik DDS dipakai; rincian ada di
[EXPERIMENTS.md](EXPERIMENTS.md).

## 4. Interpretasi

Feature importance dan SHAP digunakan untuk menjelaskan perilaku model, bukan
untuk menyatakan hubungan sebab-akibat. Hasil SHAP dilaporkan bersama model,
fitur, sampel, dan jenis explainer yang digunakan.
