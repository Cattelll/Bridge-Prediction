# Alur Eksperimen — Target `matches_par_contract`

Rangkuman alur pipeline dan **apa yang berubah di setiap eksperimen**
(`notebooks/03`–`06`). Cara kerja dan perbedaan ketiga algoritma dijelaskan
terpisah di [ALGORITHMS.md](ALGORITHMS.md).

Angka diambil dari output notebook, `outputs/results/nb05_summary.json`, dan
evaluasi ulang model tersimpan di `outputs/models/` pada test set
(run 2026-09-27, setelah definisi label diperketat).

---

## 1. Apa yang diprediksi

Untuk setiap board, model menjawab: **apakah kontrak yang dibid pemain manusia
sama dengan salah satu kontrak par menurut Double-Dummy Solver (DDS)?** Yang
dibandingkan: level, strain, status doubled/redoubled, dan sisi declarer
(NS/EW). Satu board bisa punya beberapa kontrak par berskor sama (mis. 4♥ atau
4♠ oleh EW); cocok dengan salah satunya sudah dihitung 1. (Sebelum 2026-09-27
hanya level+strain terhadap kontrak par pertama yang dibandingkan.)

| Label `matches_par_contract` | Arti | Porsi data |
|---|---|---|
| **1** | Kontrak dibid = kontrak par (mis. dibid 4♠ oleh N, par 4♠ oleh NS) | 24,3% |
| **0** | Kontrak dibid ≠ kontrak par (mis. dibid 3NT, par 4♠; atau 4♠ oleh E, par 4♠ oleh NS) | 75,7% |

Kelasnya **tidak seimbang** (sekitar 1 : 3). Model yang selalu menebak "0"
sudah mendapat accuracy 75,7%, jadi **F1 macro** dipakai sebagai metrik utama
di samping accuracy.

**Input: 174 fitur numerik per board.** Terdiri dari 96 fitur per-seat,
44 partnership, 1 HCP advantage, 8 deal context, 15 auction, dan 10 trik
double-dummy. **Kolom par DDS** (8 kolom `dd_par_*` + `dd_par_contracts`) **tidak dipakai sebagai fitur**
karena kolom itu bahan langsung pembentuk label (*leakage*).

---

## 2. Alur umum (sama untuk semua eksperimen)

```
File .lin (606) + .pbn (1.390) ──parse──► board (tangan, lelang, kontrak)
        │
        ├─► 164 fitur kanonik + 10 trik DD   = 174 fitur (X)
        └─► kontrak par DDS vs kontrak dibid = label 0/1 (y)
        │
Split 70/15/15 group-aware (open & closed room papan yang sama selalu satu split)
   train 34.829  |  val 7.462  |  test 7.464
        │
Train ──► model.fit(X_train, y_train [, sample_weight])
Val   ──► memilih hyperparameter / kandidat
Test  ──► disentuh sekali di akhir untuk angka final
        │
predict_proba(X) → peluang kelas 1 → ≥ 0,5 ⇒ "1 (cocok par)", selain itu "0"
```

**Yang sama di SEMUA eksperimen** (hanya berubah kalau disebut eksplisit):
data & split, 174 fitur, `random_state=42`, ambang prediksi 0,5, dan metrik
evaluasi.

---

## 3. Peta eksperimen

Setiap eksperimen mengubah **satu atau dua hal** dibanding pembandingnya:

```
                        ┌─► Exp A: hyperparameter XGBoost ──┬─► Exp B: + bobot kelas penuh
Baseline XGBoost ───────┘                                   └─► Exp F: + bobot kelas parsial (α)
Baseline LightGBM ──────────► Exp C: hyperparameter LightGBM
Baseline Random Forest ─────► Exp E: hyperparameter Random Forest
```

| Eksperimen | Algoritma | Pembanding | Yang diubah | Kriteria pilih (val) |
|---|---|---|---|---|
| Ablasi 1b | Random Forest | 164 fitur | **Set fitur**: +10 trik DD | F1 macro, ambang +0,5pp |
| Baseline | RF / XGB / LGBM | – | – (hyperparameter `config.yaml`) | – |
| Exp A | XGBoost | Baseline XGBoost | **Hyperparameter** (grid 4 kandidat) | Accuracy |
| Exp B | XGBoost | Exp A | **Bobot kelas**: tanpa → balanced penuh | – |
| Exp C | LightGBM | Baseline LightGBM | **Hyperparameter** (grid 4 kandidat) | F1 macro |
| Exp E | Random Forest | Baseline Random Forest | **Hyperparameter** (grid 4 kandidat) | F1 macro |
| Exp F | XGBoost | Exp A & Exp B | **Bobot kelas**: parsial, α ∈ {0,25; 0,5; 0,75} | Rata-rata accuracy + F1 macro |

---

## 4. Detail per eksperimen

Metrik "kelas 1" di bawah dihitung di test set. Porsi kelas 1 yang sebenarnya
di test set = **24,3%**; "porsi prediksi 1" menunjukkan seberapa sering model
menebak "cocok par".

### 4.1 Ablasi fitur trik DD — `notebooks/03` bagian 1b

| | Sebelum | Sesudah |
|---|---|---|
| Fitur | 164 kanonik | 164 + 10 trik DD (`ns_dd_*`, `ew_dd_*`) = 174 |
| Model | Random Forest cepat | sama |
| F1 macro (val) | 0,6956 | **0,7496** (+5,40pp) |

**Yang berubah:** hanya set fitur.
**Keputusan:** kenaikan jauh di atas ambang +0,5pp → 174 fitur dipakai untuk
semua eksperimen berikutnya.

### 4.2 Baseline resmi — `notebooks/03` → `04`

Ketiga algoritma dilatih dengan hyperparameter `configs/config.yaml` apa
adanya. Detail tiap parameter ada di [ALGORITHMS.md](ALGORITHMS.md).

| Model | Konfigurasi | Bobot kelas |
|---|---|---|
| Random Forest | 300 pohon, tanpa batas kedalaman, `min_samples_split=5`, `max_features="sqrt"` | balanced |
| XGBoost | 300 pohon, depth 6, lr 0,05, subsample 0,8, colsample 0,8 | **tidak ada** |
| LightGBM | 300 pohon, 63 leaves, lr 0,05, colsample 0,8 | balanced |

| Model | Acc | F1 macro | Recall kelas 1 | Precision kelas 1 | Porsi prediksi 1 |
|---|---|---|---|---|---|
| Random Forest | 82,3% | 0,754 | 60,3% | 64,7% | 22,6% |
| XGBoost | 83,0% | 0,729 | 44,9% | 75,3% | 14,5% |
| LightGBM | 81,5% | 0,761 | 70,1% | 60,2% | 28,3% |

**Catatan:** perbedaan ketiga baseline **bukan perbandingan yang adil
antar algoritma**, karena XGBoost satu-satunya yang tanpa bobot kelas. Itulah
sebabnya XGBoost jarang menebak "1" (14,5%) dan F1 macro-nya paling rendah.

### 4.3 Eksperimen A — XGBoost, retuning hyperparameter

**Pembanding:** baseline XGBoost.
**Yang berubah:** hyperparameter, dipilih dari 4 kandidat berdasarkan
**accuracy** val.
**Yang tetap:** tanpa bobot kelas.

| Kandidat | n_estimators | max_depth | lr | subsample | colsample | lain | Acc val | F1 macro val |
|---|---|---|---|---|---|---|---|---|
| default (config.yaml) | 300 | 6 | 0,05 | 0,8 | 0,8 | – | 0,8236 | 0,7216 |
| **deeper_more_trees** ✔ | **500** | **8** | 0,05 | **0,9** | 0,8 | `min_child_weight=1`, `reg_lambda=1,0` | **0,8423** | 0,7608 |
| mid_capacity | 400 | 7 | 0,08 | 0,85 | 0,7 | `min_child_weight=3`, `reg_lambda=1,5` | 0,8370 | 0,7507 |
| leaner_faster | 250 | 6 | 0,1 | 0,8 | 0,8 | – | 0,8313 | 0,7382 |

**Perubahan efektif vs baseline:** pohon lebih banyak (300 → 500), lebih dalam
(6 → 8), dan memakai lebih banyak baris per pohon (80% → 90%).
(`min_child_weight=1` dan `reg_lambda=1,0` sama dengan default XGBoost.)

| Test set | Acc | F1 macro | Recall kelas 1 | Precision kelas 1 | Porsi prediksi 1 |
|---|---|---|---|---|---|
| Baseline XGBoost | 83,0% | 0,729 | 44,9% | 75,3% | 14,5% |
| **Exp A** | **84,9%** | 0,769 | 53,7% | 77,1% | 16,9% |

**Dampak:** pohon yang lebih besar membuat model lebih mampu mengenali board
cocok par (recall +8,8pp) sekaligus sedikit lebih tepat (precision +1,8pp).
Tapi karena masih tanpa bobot, model tetap jarang menebak "1" (16,9% vs 24,3%
sebenarnya).
Accuracy tertinggi di seluruh proyek.

> File model: `outputs/models/xgboost_expA_default.pkl`. Nama file-nya
> menyesatkan: isinya kandidat `deeper_more_trees`, bukan default
> `config.yaml`.

### 4.4 Eksperimen B — XGBoost + bobot kelas penuh

**Pembanding:** Exp A.
**Yang berubah:** hanya bobot kelas. `sample_weight = compute_sample_weight("balanced", y_train)`
→ kelas 1 ≈ **2,06**, kelas 0 ≈ **0,66** (setara `class_weight="balanced"`
di RF/LightGBM).
**Yang tetap:** hyperparameter Exp A (500 pohon, depth 8, …).

| Test set | Acc | F1 macro | Recall kelas 1 | Precision kelas 1 | Porsi prediksi 1 |
|---|---|---|---|---|---|
| Exp A (tanpa bobot) | 84,9% | 0,769 | 53,7% | 77,1% | 16,9% |
| **Exp B** (bobot penuh) | 83,2% | 0,773 | **66,7%** | 65,0% | 25,0% |

**Dampak:** model sekarang menebak "1" kira-kira sesering proporsi aslinya
(25,0%). Recall kelas 1 naik +13,0pp, tapi precision turun −12,1pp karena lebih banyak
board kelas 0 salah ditebak "1". Hasilnya F1 macro naik, accuracy turun.

### 4.5 Eksperimen C — LightGBM, retuning hyperparameter

**Pembanding:** baseline LightGBM.
**Yang berubah:** hyperparameter, dipilih dari 4 kandidat berdasarkan
**F1 macro** val.
**Yang tetap:** `class_weight="balanced"`.

| Kandidat | n_estimators | num_leaves | max_depth | lr | subsample* | colsample | Acc val | F1 macro val |
|---|---|---|---|---|---|---|---|---|
| default (config.yaml) | 300 | 63 | -1 | 0,05 | 0,8 | 0,8 | 0,8140 | 0,7605 |
| more_leaves | 400 | 95 | -1 | 0,05 | 0,8 | 0,8 | 0,8282 | 0,7715 |
| lower_lr_more_trees | 500 | 63 | -1 | 0,03 | 0,85 | 0,7 | 0,8147 | 0,7611 |
| **deep_constrained** ✔ | **350** | **127** | **9** | 0,05 | 0,8 | 0,8 | 0,8281 | **0,7726** |

\* `subsample` tidak berpengaruh di LightGBM karena `subsample_freq=0`
(lihat [ALGORITHMS.md](ALGORITHMS.md) §4).

**Perubahan efektif vs baseline:** lebih banyak pohon (300 → 350), pohon jauh
lebih besar (63 → 127 daun), tapi kedalamannya dibatasi (tanpa batas → 9).
`more_leaves` (95 daun) nyaris sama baiknya (selisih 0,1pp di val).

| Test set | Acc | F1 macro | Recall kelas 1 | Precision kelas 1 | Porsi prediksi 1 |
|---|---|---|---|---|---|
| Baseline LightGBM | 81,5% | 0,761 | 70,1% | 60,2% | 28,3% |
| **Exp C** | 83,1% | 0,775 | 68,4% | 64,2% | 25,9% |

**Dampak:** model sedikit lebih "yakin". Porsi tebakan "1" turun mendekati
proporsi asli, precision naik +4,0pp dengan recall hanya turun −1,7pp. Hasilnya
accuracy +1,6pp dan F1 macro +1,4pp.

### 4.6 Eksperimen E — Random Forest, retuning hyperparameter

**Pembanding:** baseline Random Forest.
**Yang berubah:** hyperparameter, dipilih dari 4 kandidat berdasarkan
**F1 macro** val.
**Yang tetap:** `class_weight="balanced"`.

| Kandidat | n_estimators | max_depth | min_samples_split | min_samples_leaf | max_features | Acc val | F1 macro val |
|---|---|---|---|---|---|---|---|
| default (config.yaml) | 300 | None | 5 | 1 | sqrt (≈13) | 0,8206 | 0,7525 |
| more_trees | 600 | None | 5 | 1 | sqrt | 0,8220 | 0,7536 |
| regularized_shallow | 400 | 25 | 10 | 4 | sqrt | 0,8167 | 0,7558 |
| **more_features** ✔ | **400** | None | 5 | 1 | **0,5 (87 fitur)** | 0,8285 | **0,7668** |

**Perubahan efektif vs baseline:** di setiap split, pohon mempertimbangkan
87 fitur acak (bukan 13), dan jumlah pohon 300 → 400. Menambah pohon saja
(`more_trees`) tidak membantu, jadi kenaikannya datang dari `max_features`.
Konsekuensinya training jauh lebih lambat (≈331 s vs ≈8 s di val).

| Test set | Acc | F1 macro | Recall kelas 1 | Precision kelas 1 | Porsi prediksi 1 |
|---|---|---|---|---|---|
| Baseline RF | 82,3% | 0,754 | 60,3% | 64,7% | 22,6% |
| **Exp E** | 82,9% | 0,766 | 63,8% | 65,2% | 23,8% |

**Dampak:** dengan lebih banyak fitur per split, tiap pohon lebih sering
menemukan fitur yang benar-benar informatif. Recall kelas 1 naik +3,5pp
dengan precision sedikit naik (+0,5pp).

### 4.7 Eksperimen F — XGBoost + bobot kelas parsial

**Pembanding:** Exp A (tanpa bobot) dan Exp B (bobot penuh).
**Yang berubah:** bobot kelas diinterpolasi di antara keduanya:

```
sample_weight = 1 + α × (bobot_balanced − 1)
   α = 0   → semua bobot 1 (= Exp A)
   α = 1   → bobot balanced penuh (= Exp B)
```

**Yang tetap:** hyperparameter Exp A. Dipilih dari 3 nilai α berdasarkan
**rata-rata accuracy dan F1 macro** val.

| α | Bobot kelas 1 / kelas 0 | Acc val | F1 macro val | Rata-rata |
|---|---|---|---|---|
| 0,25 | ≈ 1,26 / 0,92 | 0,8427 | 0,7702 | 0,8065 |
| **0,50** ✔ | **≈ 1,53 / 0,83** | 0,8401 | 0,7736 | **0,8069** |
| 0,75 | ≈ 1,79 / 0,75 | 0,8338 | 0,7728 | 0,8033 |

| Test set | Acc | F1 macro | Recall kelas 1 | Precision kelas 1 | Porsi prediksi 1 |
|---|---|---|---|---|---|
| Exp A (α = 0) | 84,9% | 0,769 | 53,7% | 77,1% | 16,9% |
| **Exp F (α = 0,5)** | 84,7% | **0,782** | 61,7% | 71,4% | 21,0% |
| Exp B (α = 1) | 83,2% | 0,773 | 66,7% | 65,0% | 25,0% |

**Dampak:** Exp F berada di tengah antara A dan B dalam hal seberapa sering
menebak "1". Recall naik +8,0pp dibanding A dengan precision tetap jauh di
atas B. Hasilnya F1 macro tertinggi, dan accuracy hanya −0,2pp dari Exp A.

---

## 5. Hasil akhir test set — `notebooks/06` (8 kandidat)

| Model | Acc | F1 macro | F1 weighted | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost Exp F (α=0,5)** | 84,7% | **0,782** | **0,843** | **0,849** | 0,707 |
| LightGBM Exp C (retuned) | 83,1% | 0,775 | 0,832 | **0,849** | **0,712** |
| XGBoost Exp B (bobot penuh) | 83,2% | 0,773 | 0,832 | 0,847 | 0,709 |
| XGBoost Exp A (retuned) | **84,9%** | 0,769 | 0,839 | 0,848 | 0,709 |
| Random Forest Exp E (retuned) | 82,9% | 0,766 | 0,829 | 0,840 | 0,700 |
| LightGBM baseline | 81,5% | 0,761 | 0,819 | 0,845 | 0,702 |
| Random Forest baseline | 82,3% | 0,754 | 0,821 | 0,823 | 0,667 |
| XGBoost baseline | 83,0% | 0,729 | 0,814 | 0,831 | 0,677 |

Pembanding: tebak mayoritas (selalu "0") = 75,7% accuracy. ROC-AUC
0,82–0,85 (bukan mendekati 1,0) menandakan tidak ada leakage yang tersisa.

Baseline resmi (`outputs/models/{randomforest,xgboost,lightgbm}.pkl`) **tidak
diganti** oleh hasil `notebooks/05`. Promosi Exp F belum diputuskan.

---

## 6. Inti temuan

1. **Bobot kelas adalah faktor penentu terbesar.** Ia mengatur seberapa sering
   model berani menebak "cocok par":
   - **Tanpa bobot** (XGBoost baseline, Exp A): menebak "1" hanya 14–17%
     dari board. Precision tinggi, recall rendah, F1 macro rendah.
   - **Bobot penuh** (Exp B, RF/LightGBM balanced): menebak "1" 22–28%.
     Recall tinggi, precision lebih rendah.
   - **Bobot setengah** (Exp F, α=0,5): menebak "1" 21,0%. Kompromi terbaik,
     F1 macro tertinggi dengan accuracy nyaris tertinggi.
2. **Retuning hyperparameter membantu, tapi lebih kecil.** Di validation set,
   F1 macro naik +1,2pp untuk LightGBM (Exp C) dan +1,4pp untuk Random Forest
   (Exp E) dibanding default `config.yaml`. Untuk XGBoost, retuning (Exp A)
   menaikkan F1 macro +3,9pp, lalu bobot parsial (Exp F) menambah +1,3pp lagi.
3. **Setelah kondisinya disamakan, ketiga algoritma berdekatan.** Kandidat
   terbaik tiap algoritma (Exp F, C, E) berada dalam rentang F1 macro
   0,766–0,782 dan ROC-AUC 0,840–0,849 di test set. Keunggulan XGBoost Exp F
   (+0,7pp F1 macro atas LightGBM Exp C) kecil dan sebagian besar berasal dari
   pengaturan bobot kelas.
