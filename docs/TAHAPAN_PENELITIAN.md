# Tahapan Penelitian

Dokumen ini menjelaskan tahapan penelitian klasifikasi kecocokan kontrak par
pada permainan Contract Bridge berdasarkan diagram alur penelitian. Penelitian
ini membandingkan tiga algoritma ensemble berbasis pohon, yaitu **Random
Forest**, **XGBoost**, dan **LightGBM**, untuk mengklasifikasikan apakah
kontrak yang dibid pemain manusia sesuai dengan salah satu kontrak par yang
dihasilkan oleh Double-Dummy Solver (DDS).

## 1. Mulai

Penelitian dimulai dengan menetapkan tujuan, ruang lingkup, sumber data,
target klasifikasi, serta metode evaluasi.

Tujuan penelitian adalah:

1. membangun dataset dari rekaman permainan Contract Bridge;
2. menghitung kontrak par berdasarkan informasi seluruh kartu menggunakan DDS;
3. membentuk target biner `matches_par_contract`;
4. melatih dan membandingkan Random Forest, XGBoost, dan LightGBM; dan
5. menganalisis hasil klasifikasi serta kontribusi fitur menggunakan SHAP.

Penelitian hanya menggunakan model tree-based untuk data tabular. Neural
network, reinforcement learning, ensemble antar-model, dan stacking tidak
termasuk dalam ruang lingkup penelitian.

## 2. Data LIN dan PBN

Data penelitian berasal dari dua format rekaman permainan:

- **LIN (`.lin`)** dari BBO/Vugraph, yang memuat distribusi kartu, proses
  bidding, vulnerability, dealer, serta sebagian informasi permainan;
- **PBN (`.pbn`)** dari arsip pertandingan dan kejuaraan bridge, yang memuat
  informasi board, pemain, kartu, auction, dan metadata permainan.

Dataset akhir menggunakan:

- 606 file LIN;
- 1.390 file PBN;
- 49.755 board setelah deduplikasi dan pembersihan data.

Setiap board diperlakukan sebagai satu unit observasi. Data mentah disimpan di
[data/raw/](../data/raw/) dan [data/raw_pbn/](../data/raw_pbn/), sedangkan
hasil pengolahan disimpan di [data/processed/](../data/processed/).

## 3. Preprocessing

Tahap preprocessing menyiapkan data mentah agar dapat diproses secara konsisten
oleh parser dan model. Tahap ini meliputi:

1. membaca file LIN dan PBN;
2. membersihkan metadata yang tidak konsisten;
3. menangani board yang memiliki informasi kartu tidak lengkap;
4. merekonstruksi tangan yang dapat diturunkan dari 52 kartu;
5. memastikan format auction dan kontrak seragam;
6. memeriksa keunikan kartu sebelum board dikirim ke DDS; dan
7. menghapus duplikasi board serta data yang tidak valid.

Parser LIN dan PBN menghasilkan representasi board yang seragam
(`BoardRecord`). Dengan demikian, tahap ekstraksi fitur berikutnya tidak perlu
membedakan sumber file.

Validasi 52 kartu unik sangat penting karena input kartu yang rusak dapat
menyebabkan DDS gagal. Board yang tidak memenuhi validasi tidak digunakan
untuk perhitungan DDS.

## 4. Ekstraksi Data

Pada tahap ini, informasi dari setiap `BoardRecord` diekstraksi menjadi
informasi terstruktur, antara lain:

- empat tangan pemain (North, East, South, dan West);
- dealer dan vulnerability;
- urutan bidding;
- kontrak akhir yang dibid;
- declarer;
- status doubled atau redoubled;
- hasil permainan dan jumlah trick; serta
- metadata sumber board.

Ekstraksi dilakukan oleh parser dan modul feature engineering. Hasil antara
tahap ini berupa satu baris data untuk setiap board yang dapat diproses.

## 5. Perhitungan Double-Dummy Solver (DDS)

DDS menghitung kemampuan optimal setiap sisi untuk memainkan kontrak
berdasarkan seluruh kartu yang terbuka. Dalam penelitian ini, DDS digunakan
untuk dua keperluan:

1. menghasilkan fitur jumlah trick double-dummy; dan
2. menentukan satu atau beberapa kontrak par yang menjadi dasar pembentukan
   label.

Untuk setiap strain (`S`, `H`, `D`, `C`, dan `N`), dihitung jumlah trick
optimal untuk sisi NS dan EW. Hasil tersebut menghasilkan 10 fitur model:

- `ns_dd_S`, `ns_dd_H`, `ns_dd_D`, `ns_dd_C`, `ns_dd_N`;
- `ew_dd_S`, `ew_dd_H`, `ew_dd_D`, `ew_dd_C`, `ew_dd_N`.

DDS juga menghasilkan informasi kontrak par, termasuk level, strain, status
double/redouble, sisi declarer, skor, dan seluruh kontrak par yang memiliki
skor sama. Jika terdapat beberapa kontrak par yang setara, semuanya
dipertahankan dan dipisahkan dalam kolom `dd_par_contracts`.

Kolom par DDS digunakan untuk membentuk label, bukan sebagai fitur input model.
Pemisahan ini mencegah kebocoran data (*data leakage*).

## 6. Pembentukan Fitur

Fitur dibentuk dari informasi kartu, konteks permainan, dan auction. Dataset
model akhir berisi 174 fitur:

| Kelompok fitur | Jumlah |
|---|---:|
| Fitur per seat (N/E/S/W) | 96 |
| Fitur partnership NS/EW | 44 |
| Keunggulan HCP NS terhadap EW | 1 |
| Dealer dan vulnerability | 8 |
| Informasi auction | 15 |
| Trick DDS per strain | 10 |
| **Total** | **174** |

Contoh fitur yang dibentuk adalah HCP, panjang setiap suit, stopper,
distribution, balanced hand, fit partnership, jumlah bid, panjang auction,
opening bid, serta jumlah trick DDS.

Fitur yang secara langsung membentuk label tidak dimasukkan ke model, yaitu:

- `dd_par_level`;
- `dd_par_denom_S`, `dd_par_denom_H`, `dd_par_denom_D`,
  `dd_par_denom_C`, dan `dd_par_denom_N`;
- `dd_par_score`;
- `dd_par_declarer_is_ns`; dan
- `dd_par_contracts`.

Sepuluh fitur trick DDS tetap digunakan karena hanya menyatakan kemampuan
trick-taking dan tidak secara langsung memberikan kontrak par atau label.
Ablasi pada pipeline menunjukkan bahwa fitur-fitur tersebut meningkatkan F1
macro validasi sekitar 5,40 poin persentase dibandingkan 164 fitur kanonik
saja.

## 7. Pembentukan Label

Target utama penelitian adalah `matches_par_contract`, yaitu target biner:

| Nilai | Makna |
|---:|---|
| 1 | Kontrak aktual yang dibid cocok dengan salah satu kontrak par DDS |
| 0 | Kontrak aktual yang dibid tidak cocok dengan kontrak par DDS |

Kecocokan kontrak diperiksa berdasarkan:

- level kontrak;
- strain atau denomination;
- status doubled/redoubled; dan
- sisi declarer, yaitu NS atau EW.

Kontrak aktual dapat memuat informasi doubled atau redoubled. Sebuah board
diberi label 1 apabila kontrak aktual cocok dengan **salah satu** kontrak par
yang tersedia, bukan hanya kontrak par pertama. Kontrak `PASS` juga ditangani
sebagai kemungkinan hasil par.

Kolom `target_base`, `target`, dan `target_category` tetap disimpan untuk
analisis dan kompatibilitas historis, tetapi bukan target utama pelatihan.

## 8. Pembentukan Dataset

Setelah fitur dan label terbentuk, seluruh board digabungkan menjadi dataset
model. Proses ini meliputi:

1. deduplikasi baris;
2. pemeriksaan nilai yang hilang dan tipe data;
3. pemisahan metadata dari kolom fitur;
4. penyimpanan urutan fitur pada `feature_columns.json`;
5. penyimpanan encoder label; dan
6. pemeriksaan distribusi kelas.

Distribusi target pada dataset akhir adalah sekitar 24,3% board yang cocok
dengan kontrak par dan 75,7% board yang tidak cocok. Ketidakseimbangan ini
menjadi pertimbangan utama dalam pemilihan metrik dan konfigurasi model.

## 9. Split Data

Dataset dibagi menjadi:

- 70% data training;
- 15% data validasi; dan
- 15% data testing.

Pembagian menggunakan `random_state=42` dan strategi
**group-aware stratified split**. Grup dibentuk berdasarkan identitas papan
fisik, yaitu kombinasi `_source_file` dan `_board_number`.

Strategi tersebut mencegah pasangan open room dan closed room yang memiliki
distribusi kartu identik tersebar ke split yang berbeda. Tanpa pengelompokan
ini, model dapat melihat board yang sama atau sangat mirip pada data training
dan testing sehingga metrik menjadi terlalu optimistis.

Data testing hanya digunakan pada evaluasi akhir. Pemilihan konfigurasi dan
kandidat eksperimen dilakukan menggunakan data training dan validasi.

## 10. Pelatihan Model

Tiga model dilatih menggunakan fitur dan split yang sama:

### 10.1 Random Forest

Random Forest membangun banyak decision tree menggunakan sampel dan subset
fitur yang berbeda. Prediksi akhir diperoleh dari agregasi seluruh tree.
Model ini digunakan sebagai metode bagging dan pembanding dasar.

### 10.2 XGBoost

XGBoost membangun tree secara bertahap dengan memperbaiki kesalahan prediksi
tree sebelumnya. Model ini digunakan sebagai metode boosting berbasis
gradient boosting.

### 10.3 LightGBM

LightGBM juga menggunakan gradient boosting, tetapi membangun tree secara
leaf-wise sehingga efisien pada dataset tabular berukuran besar.

Hyperparameter baseline diambil dari
[configs/config.yaml](../configs/config.yaml). Model baseline tidak
menggunakan kolom par DDS yang membentuk label.

## 11. Evaluasi dan Perbandingan

Setelah pelatihan, setiap model menghasilkan prediksi kelas dan probabilitas
kelas 1. Evaluasi dilakukan pada data test menggunakan:

- accuracy;
- precision dan recall kelas positif;
- F1 score kelas positif;
- F1 macro;
- F1 weighted;
- ROC-AUC; dan
- PR-AUC (*average precision*).

F1 macro menjadi metrik penting karena distribusi kelas tidak seimbang.
Accuracy tetap dilaporkan, tetapi tidak digunakan sebagai satu-satunya dasar
pemilihan model karena baseline mayoritas sudah menghasilkan accuracy sekitar
75,7%.

Hasil baseline pada test set:

| Model | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 82,3% | 0,754 | 0,821 | 0,823 | 0,667 |
| XGBoost | **83,0%** | 0,729 | 0,814 | 0,831 | 0,677 |
| LightGBM | 81,5% | **0,761** | 0,819 | **0,845** | **0,702** |

Selain baseline, eksperimen retuning dan pembobotan kelas dilakukan pada
[notebooks/05_improvement_experiments.ipynb](../notebooks/05_improvement_experiments.ipynb)
dan dibandingkan kembali pada
[notebooks/06_final_evaluation.ipynb](../notebooks/06_final_evaluation.ipynb).
Kandidat terbaik keseluruhan adalah XGBoost soft-balanced dengan `alpha=0,5`,
yang memperoleh accuracy 84,7% dan F1 macro 0,782 pada evaluasi final.

## 12. Analisis SHAP

SHAP (*SHapley Additive exPlanations*) digunakan untuk menjelaskan kontribusi
fitur terhadap prediksi model. Analisis ini menjawab:

- fitur apa yang paling berpengaruh terhadap keputusan model;
- apakah suatu fitur mendorong prediksi ke kelas cocok atau tidak cocok;
- apakah pengaruh fitur konsisten untuk berbagai board; dan
- apakah terdapat indikasi fitur yang terlalu dekat dengan target.

Analisis SHAP dilakukan setelah model dilatih dan dievaluasi. Kolom par DDS
yang digunakan untuk membentuk label tetap tidak boleh dianalisis sebagai
fitur model karena kolom tersebut memang dikeluarkan dari input untuk
mencegah leakage.

## 13. Analisis Hasil

Tahap analisis hasil menyatukan metrik, confusion matrix, hasil eksperimen,
dan analisis SHAP. Pembahasan mencakup:

1. perbandingan kemampuan tiga baseline;
2. kemampuan model mengenali kelas positif yang lebih jarang;
3. pengaruh fitur trick DDS terhadap performa;
4. dampak pembobotan kelas dan retuning hyperparameter;
5. trade-off accuracy, F1 macro, ROC-AUC, dan PR-AUC;
6. fitur dominan berdasarkan SHAP;
7. pemeriksaan indikasi kebocoran data; dan
8. keterbatasan penelitian.

Berdasarkan hasil saat ini, LightGBM merupakan baseline resmi terbaik untuk
F1 macro, ROC-AUC, dan PR-AUC. XGBoost soft-balanced merupakan kandidat
eksperimen terbaik secara keseluruhan karena memperoleh F1 macro tertinggi
di antara kandidat retuning dan accuracy yang tetap tinggi. Kandidat tersebut
belum menggantikan baseline resmi karena baseline dan eksperimen dipisahkan
untuk menjaga keterlacakan evaluasi.

## 14. Selesai

Penelitian selesai setelah seluruh tahapan berikut terdokumentasi dan dapat
direproduksi:

- sumber dan proses preprocessing data;
- aturan pembentukan fitur;
- definisi label dan pencegahan leakage;
- strategi group-aware split;
- konfigurasi dan hasil pelatihan model;
- metrik evaluasi pada data test;
- hasil analisis SHAP; dan
- kesimpulan serta batasan penelitian.

Pipeline notebook resmi dijalankan berurutan dari
[notebooks/01_data_extraction.ipynb](../notebooks/01_data_extraction.ipynb)
sampai
[notebooks/06_final_evaluation.ipynb](../notebooks/06_final_evaluation.ipynb).
Artefak hasil evaluasi tersedia di
[outputs/results/](../outputs/results/).

## Ringkasan Alur

```text
Mulai
  ↓
Data LIN dan PBN
  ↓
Preprocessing
  ↓
Ekstraksi data menjadi BoardRecord
  ↓
Perhitungan DDS
  ↓
Pembentukan fitur ─────┐
Pembentukan label ────┤
                      ↓
                   Dataset
                      ↓
                 Split data
                      ↓
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
 Random Forest      XGBoost        LightGBM
       └──────────────┼──────────────┘
                      ↓
           Evaluasi dan perbandingan
                      ↓
                Analisis SHAP
                      ↓
                Analisis hasil
                      ↓
                    Selesai
```
