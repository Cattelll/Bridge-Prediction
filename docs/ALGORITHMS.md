# Algoritma yang Digunakan: Random Forest, XGBoost, LightGBM

Penjelasan tiga algoritma klasifikasi yang dibandingkan di penelitian ini, cara
masing-masing bekerja untuk target biner `matches_par_contract`, dan apa
bedanya. Konfigurasi yang disebut di sini diambil dari `configs/config.yaml`
dan dicek langsung dari model tersimpan di `outputs/models/`
(scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0).

Alur tiap eksperimen dan apa yang berubah di masing-masing ada di
[EXPERIMENTS.md](EXPERIMENTS.md).

---

## 1. Dasar bersama: pohon keputusan

Ketiga algoritma adalah **ensemble pohon keputusan**: banyak pohon yang
hasilnya digabung. Jadi yang perlu dipahami dulu adalah cara satu pohon
mengklasifikasi.

Satu pohon keputusan mengajukan serangkaian pertanyaan ya/tidak tentang fitur,
dari akar sampai daun:

```
ns_hcp ≥ 25 ?
├─ ya → ns_best_fit ≥ 8 ?
│        ├─ ya → auction_competitive = 0 ? → daun: P(cocok par) = 0,45
│        └─ tidak → ...
└─ tidak → ...
```

*(Ilustrasi. Pertanyaan yang sebenarnya dipilih oleh algoritma.)*

- **Saat training**, di setiap simpul algoritma mencoba banyak kombinasi
  fitur + nilai batas, lalu memilih yang **paling baik memisahkan** board
  kelas 0 dan kelas 1.
- **Saat klasifikasi**, satu board menelusuri pohon sesuai nilai fiturnya sampai
  tiba di satu daun. Daun itu berisi peluang board tersebut cocok par.
- Karena semua fitur dipakai lewat perbandingan "≥ batas", pohon **tidak perlu
  normalisasi fitur** dan bisa menangkap interaksi antar fitur secara alami
  (mis. HCP tinggi *dan* fit ≥ 8).

Satu pohon saja mudah *overfitting* (terlalu hafal data training). Ketiga
algoritma mengatasinya dengan menggabungkan banyak pohon, tetapi **dengan
cara yang berbeda**.

---

## 2. Random Forest — bagging

**Ide:** buat banyak pohon yang *berbeda-beda secara acak*, lalu ambil
rata-ratanya. Kesalahan acak tiap pohon saling menutupi.

**Cara training:**
1. Untuk tiap pohon, ambil sampel acak baris training **dengan pengembalian**
   (*bootstrap*), sebanyak jumlah data training.
2. Tumbuhkan pohon sedalam mungkin (`max_depth=None`). Di setiap simpul, hanya
   **sebagian fitur acak** yang boleh dipertimbangkan (`max_features`).
3. Ulangi sampai 300 pohon (baseline). Semua pohon **independen**, jadi bisa
   dilatih paralel.

**Cara klasifikasi:** tiap pohon memberi peluang kelas 1, lalu **dirata-rata**.
Kalau rata-ratanya ≥ 0,5, board diklasifikasikan "cocok par".

**Pemilihan pertanyaan terbaik:** memakai **Gini impurity**, yaitu ukuran
seberapa campur kelas 0 dan 1 di satu simpul. Split terbaik adalah yang paling
menurunkan Gini.

**Penanganan ketidakseimbangan kelas:** `class_weight="balanced"`. Setiap board
diberi bobot `n_total / (2 × n_kelas)`: kelas 1 ≈ **2,06**, kelas 0 ≈ **0,66**.
Board yang cocok par jadi "lebih berat" saat menghitung Gini.

**Hyperparameter yang dipakai:**

| Parameter | Baseline | Arti |
|---|---|---|
| `n_estimators` | 300 | jumlah pohon |
| `max_depth` | None | pohon tumbuh tanpa batas kedalaman |
| `min_samples_split` | 5 | simpul baru dipecah kalau berisi ≥ 5 board |
| `max_features` | `"sqrt"` (default sklearn) | tiap split hanya melihat √174 ≈ 13 fitur acak |
| `class_weight` | balanced | bobot kelas seperti di atas |

---

## 3. XGBoost — gradient boosting

**Ide:** buat pohon **satu per satu**. Tiap pohon baru bertugas **memperbaiki
kesalahan** gabungan pohon-pohon sebelumnya.

**Cara training:**
1. Mulai dari skor awal yang sama untuk semua board (berdasarkan proporsi
   kelas 1).
2. Hitung seberapa salah skor saat ini untuk tiap board, dalam bentuk
   **gradien log-loss**: board kelas 1 yang peluangnya masih rendah punya
   "kesalahan" besar.
3. Latih pohon kecil (`max_depth=6`) untuk memodelkan kesalahan itu. Split
   dipilih yang paling menurunkan log-loss, dengan penalti regularisasi
   (`reg_lambda`) supaya nilai daun tidak ekstrem.
4. Tambahkan pohon itu ke model, tapi dikecilkan dulu dengan `learning_rate`
   (0,05). Artinya tiap pohon hanya mengoreksi 5% dari yang ia pelajari.
5. Ulangi sampai 300 pohon.

**Cara klasifikasi:** skor semua pohon **dijumlahkan**, lalu diubah jadi peluang
lewat fungsi **sigmoid**. Peluang ≥ 0,5 diklasifikasikan "cocok par".

**Pertumbuhan pohon:** *level-wise* (depth-wise). Pohon dipecah lapis demi lapis
sampai `max_depth`, jadi bentuknya relatif simetris.

**Pencarian split:** berbasis histogram (`tree_method="hist"`, default sejak
XGBoost 2.0; di model tersimpan tercatat sebagai `grow_quantile_histmaker`).
Nilai tiap fitur dikelompokkan ke sejumlah *bin*, lalu split dicari antar bin.

**Keacakan:** `subsample=0.8` (tiap pohon memakai 80% baris acak) dan
`colsample_bytree=0.8` (tiap pohon memakai 80% fitur acak).

**Penanganan ketidakseimbangan kelas:** XGBoost **tidak punya
`class_weight`**. Di baseline, XGBoost dilatih **tanpa bobot kelas**, sehingga
cenderung mengklasifikasikan ke kelas mayoritas (0). Penyeimbangan hanya bisa dilakukan
lewat `sample_weight` (bobot per baris saat `fit`), yang diuji di Eksperimen B
dan F.

**Hyperparameter yang dipakai:**

| Parameter | Baseline | Arti |
|---|---|---|
| `n_estimators` | 300 | jumlah pohon (ronde boosting) |
| `max_depth` | 6 | kedalaman maksimum tiap pohon |
| `learning_rate` | 0,05 | seberapa besar kontribusi tiap pohon |
| `subsample` | 0,8 | fraksi baris per pohon |
| `colsample_bytree` | 0,8 | fraksi fitur per pohon |
| `min_child_weight` | 1 (default) | batas minimum "bobot" di satu daun |
| `reg_lambda` | 1,0 (default) | regularisasi L2 nilai daun |
| `eval_metric` | logloss | fungsi kesalahan untuk klasifikasi biner |

---

## 4. LightGBM — gradient boosting leaf-wise

**Ide:** sama dengan XGBoost (boosting, tiap pohon memperbaiki kesalahan
sebelumnya, peluang lewat sigmoid). Bedanya terutama di **cara pohon
ditumbuhkan** dan **dukungan `class_weight`**.

**Pertumbuhan pohon: *leaf-wise*.** Di setiap langkah, LightGBM memecah
**satu daun yang paling banyak menurunkan kesalahan**, di mana pun letaknya.
Pertumbuhan dibatasi jumlah daun (`num_leaves=63`), bukan kedalaman
(`max_depth=-1` = tanpa batas). Hasilnya pohon bisa asimetris: dalam di bagian
data yang sulit, dangkal di bagian yang mudah.

```
Level-wise (XGBoost)           Leaf-wise (LightGBM)
        ●                              ●
      ┌─┴─┐                          ┌─┴─┐
      ●   ●                          ●   ○
     ┌┴┐ ┌┴┐                        ┌┴┐
     ○ ○ ○ ○                        ●   ○
                                   ┌┴┐
                                   ○ ○
```

**Pencarian split berbasis histogram**, sama seperti XGBoost versi yang
dipakai di sini. LightGBM menambahkan optimasi lain untuk kecepatan (mis.
*Exclusive Feature Bundling*, yang menggabungkan fitur-fitur jarang/one-hot
yang tidak pernah bernilai tidak-nol bersamaan).

**Penanganan ketidakseimbangan kelas:** `class_weight="balanced"`, sama seperti
Random Forest (kelas 1 ≈ 2,06, kelas 0 ≈ 0,66).

**Hyperparameter yang dipakai:**

| Parameter | Baseline | Arti |
|---|---|---|
| `n_estimators` | 300 | jumlah pohon |
| `num_leaves` | 63 | jumlah daun maksimum per pohon |
| `max_depth` | -1 | tanpa batas kedalaman (dibatasi `num_leaves`) |
| `learning_rate` | 0,05 | kontribusi tiap pohon |
| `colsample_bytree` | 0,8 | fraksi fitur per pohon |
| `subsample` | 0,8 | **tidak aktif** (lihat catatan) |
| `min_child_samples` | 20 (default) | minimum board di satu daun |
| `class_weight` | balanced | bobot kelas |

> **Catatan `subsample` di LightGBM.** Di LightGBM, `subsample` (alias
> `bagging_fraction`) baru berlaku kalau `subsample_freq` (alias
> `bagging_freq`) > 0. Semua model LightGBM di proyek ini memakai
> `subsample_freq=0` (default), jadi **setiap pohon tetap memakai 100% baris
> training**. Nilai `subsample=0.8` di `config.yaml` tidak berpengaruh. Ini
> berbeda dari XGBoost, di mana `subsample=0.8` langsung aktif.

---

## 5. Perbandingan langsung

| Aspek | Random Forest | XGBoost | LightGBM |
|---|---|---|---|
| Paradigma | **Bagging** | **Boosting** | **Boosting** |
| Hubungan antar pohon | Independen, paralel | Berurutan, tiap pohon memperbaiki sebelumnya | Berurutan, tiap pohon memperbaiki sebelumnya |
| Bentuk pohon | Dalam, tanpa batas | Dangkal, level-wise (`max_depth` 6) | Leaf-wise, dibatasi `num_leaves` (63) |
| Penggabungan output pohon | Rata-rata peluang | Jumlah skor → sigmoid | Jumlah skor → sigmoid |
| Kriteria split | Gini impurity, nilai asli | Penurunan log-loss + regularisasi L2, histogram | Penurunan log-loss, histogram |
| Keacakan baris | Bootstrap per pohon | `subsample` 0,8 per pohon | Tidak aktif (lihat catatan) |
| Keacakan fitur | Per split (√174 ≈ 13 fitur) | Per pohon (80%) | Per pohon (80%) |
| `learning_rate` | Tidak ada | 0,05 | 0,05 |
| Regularisasi eksplisit | Tidak (hanya lewat keacakan & `min_samples_split`) | `reg_lambda`, `min_child_weight` | `min_child_samples` |
| Bobot kelas di baseline | `class_weight` balanced | **Tidak ada** | `class_weight` balanced |
| Waktu training baseline* | ~9–11 s | ~3 s | ~2–3 s |

\* Dari `notebooks/03` (dua kali pengukuran di sel berbeda), di mesin proyek,
run 2026-09-27. Satu fit XGBoost di run itu sempat tercatat ~1.071 s; sel
pengukuran waktu khusus mencatat 3,0 s, jadi angka besar itu kemungkinan
kontensi CPU sesaat (lihat catatan keandalan di CLAUDE.md), bukan sifat
algoritmanya.

### Hasil run terbaru (test set, baseline)

| Model | Accuracy | F1 macro | Recall kelas 1 | Precision kelas 1 | AP |
|---|---:|---:|---:|---:|---:|
| Random Forest | 80,32% | 0,7095 | 47,9% | 63,1% | 0,5888 |
| XGBoost | 80,89% | 0,6947 | 40,2% | 69,1% | 0,5946 |
| LightGBM | 79,16% | 0,7295 | 63,7% | 56,7% | 0,6285 |

Prevalensi kelas 1 pada test sekitar 24,5%. AP adalah Average Precision (skor
`average_precision_score`), bukan luas trapezoid interpolasi PR. Angka ini berasal
dari split deal-aware terbaru (50.227 baris); angka run sebelum 2026-10-16 tidak
dipakai sebagai hasil final karena deal yang sama dapat menyeberang split.

Pemilihan utama kandidat retuned dilakukan di validation berdasarkan AP: Random Forest Exp E tertinggi.
LightGBM Exp C tertinggi pada F1 macro sekunder; XGBoost Exp A tertinggi pada accuracy. Tabel lengkap dan nilai test
kandidat tersebut ada di [EXPERIMENTS.md](EXPERIMENTS.md). Jangan memilih model
dari peringkat test.

### Ringkasan singkat perbedaannya

- **Random Forest vs boosting:** Random Forest mengurangi *variance* dengan
  merata-rata banyak pohon kuat yang independen. XGBoost/LightGBM mengurangi
  *bias* dengan menumpuk banyak pohon lemah yang saling mengoreksi. Boosting
  umumnya lebih akurat, tetapi lebih sensitif terhadap hyperparameter.
- **XGBoost vs LightGBM:** algoritma dasarnya sama, dan di versi yang dipakai
  keduanya mencari split lewat histogram. Bedanya di bentuk pohon (level-wise
  vs leaf-wise), regularisasi default (XGBoost `reg_lambda=1`, LightGBM 0),
  dan dukungan `class_weight` (hanya LightGBM). Di proyek ini perbedaan hasil
  terbesar keduanya datang dari **bobot kelas**, bukan dari bentuk pohon.
