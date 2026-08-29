# CLAUDE.md — Bridge Contract Prediction

## Ringkasan Proyek
Penelitian skripsi membandingkan tiga algoritma ensemble berbasis pohon
(Random Forest, XGBoost, LightGBM) untuk memprediksi kontrak optimal
Contract Bridge dari rekaman BBO LIN format.

## Status Proyek (Juli–Agustus 2026)
Parser, feature engineering, training, dan evaluasi sudah diimplementasi.
`notebooks/01-06` adalah pipeline resmi saat ini: 606 file `.lin` + 1.390
file `.pbn` gabungan, 182 fitur (164 kanonik + 18 DDS), **49.755 board**,
**dieksekusi ulang penuh 01→06 pada 2026-08-27** (kernel `bridge-venv`) —
semua notebook lolos "Restart & Run All" berurutan tanpa error;
`data/processed/`, `outputs/models/*.pkl`, `outputs/results/*` (termasuk
`nb04_summary.json` yang dipakai `scripts/report.py`) semuanya konsisten
dan sinkron. **Perubahan besar 2026-08-27**: XGBoost baseline resmi
dipromosikan ke default `configs/config.yaml` (57.5% acc / 0.390 F1 macro,
naik dari 56.1%/0.342); nb05 & nb06 diperbaiki sehingga kembali
reproducible. Detail: "Audit menyeluruh notebooks/01–06" di bawah.
**Catatan**: `scripts/run_pipeline.py` **belum diupdate** dan masih
membangun pipeline 164-fitur/LIN-only yang lama — lihat peringatan di
bawah, jangan dijalankan tanpa diupdate dulu. `docs/SUMMARY.md` juga
masih mendeskripsikan pipeline lama (10.223 board/164 fitur) — belum
disinkronkan.

**Perbaikan metodologi split (2026-07-09, sore)**: ditemukan kebocoran
data — BBO vugraph mencatat tiap papan dua kali (open/closed room, kartu
identik), dan `train_test_split` acak biasa membiarkan ~46% pasangan itu
terpecah ke split berbeda (~60% baris val/test punya "kembaran" tangan
identik di train). `build_dataset()` di `src/preprocessing/dataset_builder.py`
sekarang pakai `StratifiedGroupKFold` dikelompokkan per papan fisik
(`_source_file` + `_board_number`) supaya pasangan open/closed room selalu
satu split. Dataset & model di atas sudah pakai split yang diperbaiki ini
(rebuild + retrain 2026-07-09). Detail investigasi & angka before/after ada
di [experiments/2026-07-09/README.md](experiments/2026-07-09/README.md).
Eksperimen lanjutan (tuning, SMOTE, two-stage, feature engineering, boosting
alternatif) ada di [experiments/2026-07-09/](experiments/2026-07-09/) —
belum pernah di-retrain di atas split yang diperbaiki ini kecuali disebutkan
eksplisit sudah diulang.

**Fitur Double-Dummy Solver + pipeline paralel (2026-07-15)**: fitur DDS
(`src/features/dds.py`, `endplay`) disetujui masuk scope sebagai fitur
TAMBAHAN opsional (lihat "Batas Ruang Lingkup" di bawah), setelah analisis
menunjukkan akurasi sudah mendekati batas konsistensi bidding manusia
sendiri (37.6% pasangan open/closed-room BBO sepakat kontrak sama persis —
`experiments/2026-07-15/README.md`). Pipeline lengkap paralel di
[`notebooks_dds/`](notebooks_dds/) + `data/processed_dds/` (182 fitur =
164 kanonik + 18 DDS), dieksekusi penuh 01→04, **tidak mengubah** pipeline
164-fitur kanonik di atas. Hasil: XGBoost dengan hyperparameter di-tune
khusus untuk 182 fitur (`n_estimators=300, max_depth=5, learning_rate=0.03,
subsample=0.9, colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0`)
mencapai **52.7% accuracy test set** (naik dari 52.1% baseline 164-fitur)
— kandidat terbaik proyek sejauh ini. Detail lengkap di
[experiments/2026-07-15/README.md](experiments/2026-07-15/README.md).

**Perluasan data — non-BBO PBN + file baru (2026-07-15, lanjutan)**:
diminta bikin crawler BBO skala besar (>=1 juta board) — ditolak, ToS BBO
eksplisit melarang scraping/automation software (`news.bridgebase.com/terms`).
Sebagai gantinya: (1) `data/raw/` bertambah dari 506 → **606 file .lin**
(100 file baru muncul selama sesi ini, kemungkinan besar export manual
pribadi dari BBO — legitimate); (2) ditambahkan **1.314 board dari 7 final
kejuaraan dunia** (Bermuda Bowl, Venice Cup, Vanderbilt, Spingold, dll.)
via `computerbridge.se`, format PBN, non-komersial — dua sumber PBN lain
yang muncul di pencarian (`bridgetoernooi.com`, Vugraph Project) ternyata
sudah mati/di-squat, jangan diulang tanpa verifikasi langsung dulu.
`src/parser/pbn_parser.py` (`PBNParser`) baru: parser PBN yang hasilnya
kompatibel dengan `BoardRecord`/`Hand` LIN, dipakai lewat parameter baru
`extra_boards` di `build_dataset()`. Dataset gabungan di
`data/processed_combined/` (606 LIN + 43 PBN, 182 fitur, **13.582 board**
total, group-aware split, tidak mengubah `data/processed/` atau
`data/processed_dds/`). **Hasil terbaik proyek sejauh ini**: LightGBM +
`class_weight="balanced"` di data gabungan → **53.6% accuracy, F1 macro
0.310, F1 weighted 0.506** (naik dari 52.6%/0.307/0.485 di 10.223 board).
Detail lengkap di
[experiments/2026-07-15/05_combined_data_source.ipynb](experiments/2026-07-15/05_combined_data_source.ipynb).

**Perluasan PBN lanjutan — tistis.nl + perbaikan 3 bug parser
(2026-07-16)**: diminta menambah lebih banyak file PBN. Ditemukan
`tistis.nl/pbn/pbn_databases.htm`, hidup dan berisi bidding manusia
terverifikasi — 169 file PBN tambahan (ETC97 1997, EYC98 1998, ETC99
1999, EYC00 2000, IWBC Boston finals, OKB, Papi Garozzo OKbridge games),
`data/raw_pbn/` naik dari 43 → **212 file** (10.223 board PBN total).
Sumber yang DITOLAK setelah verifikasi kandungan bot: `bermuda2000.zip`
(100% GIB vs WBridge5, tanpa manusia), file "Al Howard" (GIB bermain 3
dari 4 kursi), `FFT_9901.zip` (satu pasangan "Bristol GIB" di antara 16
pasangan — kontaminasi kecil tapi tidak sepadan untuk 84 board).
Integrasi mengungkap **3 bug tersembunyi**, semua diperbaiki: (1) PBN
`"#"` (shorthand "sama seperti board sebelumnya") tidak di-resolve →
identity key salah untuk tag seperti `HomeTeam`/`VisitTeam`, diperbaiki
dengan carry-forward dict per file di `PBNParser`; (2) turnamen
round-robin memakai ulang nomor board 1..32 di tiap match yang main
simultan, nomor board mentah saja bukan identity key unik → diperbaiki
dengan composite key `board|home|visit` (fallback ke nama pemain kalau
tanpa tag tim); (3) kolom `_room` kosong (`""`) berubah jadi `NaN` lewat
CSV round-trip, lalu `.astype(str)` menghasilkan string `"nan"` (bukan
`""`) → merusak merge DDS cache, diperbaiki dengan
`keep_default_na=False` saat load cache. Dataset gabungan dibangun ulang
penuh (termasuk DDS recompute penuh, tidak reuse cache lama — reuse
posisional diverifikasi TIDAK valid setelah bug fix): **21.675 board**
(naik dari 13.582, +60%), 36 kelas (naik dari 35 — `5N` akhirnya cukup
sampel), 182 fitur. **Hasil terbaik proyek — baru lagi**: LightGBM +
`class_weight="balanced"` → **54.3% accuracy, F1 macro 0.349, F1
weighted 0.519** (naik dari 53.6%/0.310/0.506 di 13.582 board), sekarang
unggul XGBoost di SEMUA metrik. Detail lengkap di
[experiments/2026-07-15/06_combined_data_v2_more_pbn.ipynb](experiments/2026-07-15/06_combined_data_v2_more_pbn.ipynb).

**Promosi data gabungan ke pipeline resmi `notebooks_dds/` (2026-07-16,
lanjutan)**: diminta mengganti data di notebook DDS dengan data baru.
`notebooks_dds/01_data_extraction.ipynb` sekarang mem-parse `data/raw/`
(606 LIN) + `data/raw_pbn/` (212 PBN) sekaligus (`PBNParser` + parameter
`extra_boards`), sehingga `data/processed_dds/` **tidak lagi** LIN-only
10.223 board — sekarang berisi data gabungan yang sama seperti
`data/processed_combined/` (**21.675 board, 36 kelas, 182 fitur**).
`notebooks_dds/03_modeling.ipynb` diupdate: LightGBM sekarang eksplisit
pakai `class_weight="balanced"` (sebelumnya default). Notebook 01→04
dieksekusi ulang penuh. **Hasil resmi pipeline `notebooks_dds/` (test
set)**: LightGBM 54.3% accuracy / F1 macro 0.349 / F1 weighted 0.519 —
unggul XGBoost (54.0%/0.294/0.504) di SEMUA metrik, angkanya identik
dengan hasil eksperimen di atas (memverifikasi pipeline resmi dan
eksperimen konsisten). `data/processed/` (164-fitur kanonik, notebooks/
biasa) **tidak diubah** — tetap terpisah sesuai desain awal.

**Perluasan besar — arsip `angelfire.com` via Wayback Machine (2026-07-16,
lanjutan)**: diminta cari lagi data PBN karena terbukti efektif menaikkan
akurasi. Ditemukan link eksternal di `tistis.nl` menunjuk ke
`angelfire.com/games2/pbnarchive/pbn/` — situs aslinya sudah **mati**
(DNS `www.angelfire.com` tidak resolve, `angelfire.com` root menolak
koneksi), tapi seluruh isinya (57 arsip zip) berhasil direcover lewat
Wayback Machine (snapshot 2019-08-06), 56/57 berhasil diunduh utuh
(1 file, `capgem00.zip`, snapshot-nya sendiri korup/truncated di
Wayback — tidak bisa direcover, dilewati). Isi: 57 arsip kejuaraan dunia
1996-2002 (Bermuda Bowl, Venice Cup, Vanderbilt, World Bridge Team
Olympiad, Cap Gemini, Cavendish Invitational, Dutch Teams Final,
European Team Championships, ACBL International Team Trials, dll.) —
3 di antaranya (`etc99`, `eyc98`, `eyc00`) ternyata **byte-identik**
dengan file yang sudah dimiliki dari `tistis.nl` (sumber sama, dua
mirror), dikeluarkan untuk mencegah duplikasi. Setelah verifikasi (cek
player-name tags untuk GIB/WBridge5/bot — **bersih**, tidak ada
kontaminasi AI), **1.178 file PBN baru** ditambahkan ke `data/raw_pbn/`
(43 computerbridge.se + 169 tistis.nl + 1.178 angelfire.com = 1.390
file total, plus 606 file LIN).

Selama komputasi DDS untuk board baru ini (37.489 board), ditemukan
**bug crash serius**: 53 board dari arsip lama (Dutch Teams Final
1996/1998, ETC 2001, WC98, Politiken 1997) punya kartu duplikat/hilang
akibat kesalahan transkripsi manual era 1990-2000an — tiap tangan
individual tetap terhitung 13 kartu, tapi total dek gabungan cuma
44-49 kartu unik (bukan 52). Memberi input ini ke `endplay`'s DDS solver
menyebabkan **segfault** (bukan exception biasa, meng-crash seluruh
proses Python) — inilah penyebab dua percobaan komputasi DDS pertama
gagal (deadlock multiprocessing dan segfault langsung). Diperbaiki
dengan validasi 52-kartu-unik di
`src/features/dds.py::compute_dds_features()` sebelum memanggil
`endplay`, mengembalikan `None` alih-alih crash untuk board semacam
ini — permanen melindungi SEMUA sumber (LIN maupun PBN), bukan cuma
yang baru. Komputasi DDS akhirnya selesai sekuensial (single-process,
karena `multiprocessing.Pool` sempat deadlock tanpa sebab jelas di
mesin ini — pelajaran: prioritaskan keandalan atas kecepatan untuk
komputasi panjang; skrip juga ditulis ulang dengan checkpoint tiap
1.000 board setelah dua kegagalan sebelumnya kehilangan seluruh
progres). Dataset gabungan dibangun ulang: **49.755 board** (naik dari
21.675, **+130%**), 36 kelas, 182 fitur. **Hasil terbaik proyek — baru
lagi**: LightGBM + `class_weight="balanced"` → **56.4% accuracy, F1
macro 0.410, F1 weighted 0.557** (naik dari 54.3%/0.349/0.519 — F1
macro naik +6.1pp / ~17.5% relatif), tetap unggul XGBoost
(56.1%/0.342/0.532) di SEMUA metrik. Promosi ke `notebooks_dds/`
(notebook 01→04 dieksekusi ulang penuh) mengonfirmasi angka identik
dengan eksperimen. Detail lengkap di
[experiments/2026-07-15/README.md](experiments/2026-07-15/README.md).

**Konsolidasi `notebooks/` menjadi satu-satunya pipeline resmi (2026-07-17)**:
sebelumnya proyek ini punya DUA pipeline notebook paralel — `notebooks/`
(164-fitur kanonik, LIN-only, terakhir dieksekusi 2026-07-10 di dataset
lama 10.223 board) dan `notebooks_dds/` (182 fitur, LIN+PBN, terakhir
dieksekusi 2026-07-16 di 49.755 board). Atas permintaan eksplisit,
`notebooks/01-04` **ditimpa** dengan versi konsolidasi: sumber data
(606 LIN + 1.390 PBN), fitur (182 = 164 kanonik + 18 DDS), dan output
sekarang identik dengan yang sebelumnya hanya ada di `notebooks_dds/`.
`data/processed/`, `outputs/models/*.pkl`, `outputs/results/*` (semua
tidak di-git) ditimpa oleh eksekusi ulang notebook 01→04 — hasil test
set diverifikasi identik dengan angka `notebooks_dds/` di atas (LightGBM
56.4%/0.410/0.557, XGBoost 56.1%/0.342/0.532, RandomForest 46.8%/0.325/
0.485). `notebooks_dds/` **dipertahankan tanpa perubahan** sebagai arsip
historis (tidak akan dieksekusi ulang lagi ke depannya) — pipeline resmi
untuk pengembangan selanjutnya adalah `notebooks/`.

Sekaligus diperbaiki bug kosmetik di notebook 1: penghitungan jumlah
file PBN (`raw_pbn_dir.glob('*.pbn') + glob('*.PBN')`) menghitung dobel
setiap file di Windows (filesystem case-insensitive membuat kedua glob
cocok dengan file yang sama) — angka yang DITAMPILKAN sempat 2x lipat
(2.780 alih-alih 1.390), tapi ini murni bug tampilan/cetak; dataset
aktual (`build_dataset()`, lewat `PBNParser.parse_directory()` yang
hanya glob sekali) tidak pernah terpengaruh — 49.755 board tetap benar
sebelum maupun sesudah fix.

> **Peringatan penting**: `scripts/run_pipeline.py` **belum diupdate** —
> masih membangun pipeline 164-fitur/LIN-only yang lama (tanpa DDS, tanpa
> PBN). Menjalankannya akan MENIMPA BALIK `data/processed/` dan
> `outputs/models/` ke versi lama 10.223-board, menghapus hasil
> konsolidasi ini. Jangan jalankan `scripts/run_pipeline.py` sampai
> skrip ini diupdate mengikuti `notebooks/01_data_extraction.ipynb`, atau
> jalankan ulang notebook 01→04 sesudahnya untuk memulihkan.

**Notebook 5 — eksperimen peningkatan baseline (2026-07-17/18)**:
ditambahkan `notebooks/05_improvement_experiments.ipynb`, didorong oleh
temuan `experiments/2026-07-15/08_baseline_dds_default.ipynb` bahwa
hyperparameter XGBoost "acc-tuned" yang dipakai di `03_modeling.ipynb`
di-tuning di dataset 10.223 board (usang untuk 49.755 board).

**Kendala teknis (3x gagal sebelum berhasil)**: percobaan pertama pakai
`RandomizedSearchCV` dengan `n_jobs=-1` di DUA level (CV search DAN
estimator XGBoost/LightGBM sekaligus) — timeout 90 menit, lalu 40 menit
(ruang pencarian dipersempit), lalu 60 menit (`n_jobs` outer diubah ke 1)
tanpa pernah selesai. Root cause: nested multiprocessing/loky di mesin
ini silently jauh lebih lambat dari seharusnya (bukan error yang gagal
cepat) — pola yang sama dengan masalah `multiprocessing.Pool` yang sudah
tercatat untuk komputasi DDS di atas. **Diperbaiki** dengan mengganti
`RandomizedSearchCV` sepenuhnya jadi grid manual kecil (4 kandidat
XGBoost, 4 kandidat LightGBM, satu fit sekuensial per kandidat,
`n_jobs=-1` HANYA di level estimator, tanpa CV/multiprocessing bersarang
sama sekali) — selesai dalam ~35 menit. **Pelajaran umum disimpan ke
memory**: di mesin ini, jangan nest `n_jobs=-1` di outer sklearn wrapper
(`RandomizedSearchCV`, dll.) yang membungkus estimator yang sudah
paralel sendiri.

**Hasil (test set, disentuh sekali di akhir)**:

| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| XGBoost (baseline resmi, acc-tuned lama) | 56.1% | 0.342 | 0.532 |
| **LightGBM (baseline resmi, class_weight)** | 56.4% | **0.410** | **0.557** |
| **XGBoost (Exp A: hyperparameter default config.yaml)** | **57.5%** | 0.390 | 0.554 |
| XGBoost (Exp B: Exp A + sample_weight balanced) | 52.4% | 0.407 | 0.539 |
| Ensemble (XGBoost Exp A + LightGBM retuned) | 56.9% | 0.400 | 0.555 |

**Temuan kunci**: (1) Grid manual mengonfirmasi ulang temuan nb08 —
hyperparameter XGBoost "default" (persis `configs/config.yaml`, BUKAN
kandidat baru hasil tuning) mengalahkan "acc-tuned" lama di semua
metrik, dan sekarang jadi **akurasi single-model tertinggi di seluruh
proyek (57.5%)**, unggul 1.3pp dari LightGBM baseline. (2) `sample_weight`
seimbang kelas untuk XGBoost (padanan `class_weight="balanced"`, yang
tidak tersedia native di XGBoost) BUKAN "nyaris gratis" seperti di
LightGBM — accuracy anjlok 5+pp untuk kenaikan F1 macro yang lebih
kecil, jadi trade-off yang jauh lebih mahal. (3) Retuning ringan
LightGBM (grid `num_leaves`/`max_depth` lebih tinggi) tidak memberi
perbaikan berarti (F1 macro val +0.3pp, dalam rentang noise) — hyperparameter
`configs/config.yaml` LightGBM yang sudah ada ternyata sudah cukup dekat
optimal di skala 49.755 board, tidak seperti XGBoost. (4) Ensemble
soft-voting berada di antara kedua model, tidak mengalahkan LightGBM di
F1 macro maupun XGBoost baru di accuracy — tidak unggul di metrik
manapun dibanding kandidat terbaiknya masing-masing.

**Kesimpulan & rekomendasi promosi**: **tidak ada kandidat yang
mengalahkan LightGBM class_weight di F1 macro** (prioritas utama proyek
mengingat class imbalance ekstrem) — baseline resmi di `notebooks/03-04`
**tetap tidak diubah**, sesuai keputusan otomatis notebook (kriteria F1
macro, lihat cell "Simpan Model Kandidat"). Namun **XGBoost "default"
config.yaml layak dipromosikan** menggantikan hyperparameter "acc-tuned"
lama di `notebooks/03_modeling.ipynb` — perbaikan nyata & tervalidasi
test set (+1.4pp accuracy, +4.8pp F1 macro) tanpa trade-off, konsisten
dengan temuan nb08. Promosi ini **SUDAH dilakukan pada 2026-08-27** — lihat
"Audit menyeluruh notebooks/01–06" di bawah. Detail eksperimen ada di
`notebooks/05_improvement_experiments.ipynb` + `outputs/results/nb05_summary.json`/
`nb05_test_comparison.csv`.

**Perluasan notebook 5 (Eksperimen E-H) + notebook 6 evaluasi final
komprehensif (2026-07-21)**: `notebooks/05_improvement_experiments.ipynb`
ditambah 4 metode baru untuk perbandingan, dan `notebooks/06_final_evaluation.ipynb`
dibuat sebagai laporan akhir tunggal yang memuat SEMUA model (3 baseline
resmi + 8 kandidat notebook 5) sekaligus.

Metode baru di notebook 5:
- **Eksperimen E** — retuning RandomForest (belum pernah dicoba di 182
  fitur/49.755 board): kandidat terbaik (`max_features=0.5`) naik dari
  46.8%→48.9% accuracy, F1 macro 0.325→0.390 (+6.5pp) — perbaikan nyata
  tapi RF tetap jauh di bawah XGBoost/LightGBM di semua metrik.
- **Eksperimen F** — `sample_weight` XGBoost PARSIAL (blend antara tanpa
  bobot dan seimbang penuh, `alpha` di antara 0-1): `alpha=0.25` jadi
  **titik tengah terbaik** — 57.5% accuracy (nyaris sama dengan Eksperimen
  A) dengan F1 macro 0.406 (nyaris sama dengan LightGBM 0.410), jauh
  lebih baik dari Eksperimen B (bobot penuh, 52.4% acc/0.407 F1 macro).
  Trade-off accuracy-vs-F1-macro yang selama ini biner (A vs B) ternyata
  bisa dioptimalkan lebih lanjut lewat interpolasi.
- **Eksperimen G** — bobot ensemble XGBoost/LightGBM dicari (bukan 50/50
  tetap): `w_xgb=0.7` sedikit lebih baik dari 50/50 di F1 macro (0.397
  val vs 0.393), tapi tidak mengalahkan model tunggal terbaik manapun.
- **Eksperimen H** — stacking (`LogisticRegression` di atas `predict_proba`
  RF+XGBoost+LightGBM): **akurasi tertinggi dari SELURUH kandidat proyek
  (58.2% test)**, tapi F1 macro-nya (0.384) yang TERENDAH di antara semua
  kandidat non-RandomForest — stacking condong ke kelas mayoritas,
  mengorbankan kelas langka lebih dari model tunggal manapun.

**Hasil akhir komprehensif (test set, `notebooks/06_final_evaluation.ipynb`,
11 kandidat dievaluasi)** — diurutkan F1 macro:

| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| **LightGBM (baseline resmi)** | 56.4% | **0.410** | 0.557 |
| XGBoost (Exp B: balanced penuh) | 52.4% | 0.407 | 0.539 |
| XGBoost (Exp F: soft-balanced α=0.25) | **57.5%** | 0.406 | **0.560** |
| Ensemble D (50/50) | 56.9% | 0.400 | 0.555 |
| Ensemble G (bobot dicari) | 57.3% | 0.397 | 0.556 |
| LightGBM (Exp C: retuned) | 55.6% | 0.394 | 0.544 |
| RandomForest (Exp E: retuned) | 48.9% | 0.390 | 0.507 |
| XGBoost (Exp A: retuned) | 57.5% | 0.390 | 0.554 |
| XGBoost (baseline resmi) | 57.5% | 0.390 | 0.554 |
| **Stacking H** | **58.2%** | 0.384 | 0.559 |
| RandomForest (baseline resmi) | 46.8% | 0.325 | 0.485 |

> Tabel di atas: angka **setelah** promosi XGBoost default config.yaml
> (2026-08-27, lihat "Audit menyeluruh" di bawah). Baris "XGBoost (baseline
> resmi)" dulu 56.1%/0.342/0.532 ("acc-tuned lama") — sekarang identik dengan
> Exp A karena keduanya memakai default config.yaml. Sisa tabel tidak berubah
> (deterministik, model & data sama).

**Kesimpulan akhir**: **LightGBM (baseline resmi, tidak berubah) tetap
model utama proyek** — F1 macro tertinggi (prioritas utama karena class
imbalance ekstrem), sesuai semua analisis sebelumnya. Tapi **XGBoost Exp F
(soft-balanced α=0.25) adalah temuan baru yang layak dicatat**: F1 macro-nya
nyaris identik dengan LightGBM (selisih 0.37pp, dalam rentang noise) namun
unggul di accuracy (+1.1pp), F1 weighted (tertinggi dari SEMUA kandidat),
DAN top-3/top-5 accuracy (tertinggi dari semua kandidat) — kandidat
"all-around terbaik" yang sepadan untuk dipertimbangkan sebagai alternatif
resmi jika prioritas penelitian condong ke accuracy/F1-weighted, bukan
murni F1 macro. Stacking H direkomendasikan HANYA jika accuracy mentah
adalah satu-satunya prioritas (mengorbankan kelas langka paling banyak
dari semua kandidat). **Update 2026-08-27**: `outputs/models/xgboost.pkl`
resmi kini memakai default config.yaml (promosi dari "acc-tuned" — lihat
"Audit menyeluruh" di bawah); `lightgbm.pkl` & `randomforest.pkl` tidak
berubah. Exp F / Stacking H tetap rekomendasi (alternatif), bukan promosi.
Detail lengkap, seluruh grafik (confusion matrix, feature importance, SHAP,
radar chart), dan tabel mentah ada di
`notebooks/06_final_evaluation.ipynb` dan
`outputs/results/nb06_final_comparison.csv`/`nb06_summary.json`.

> **Catatan keandalan lanjutan**: eksekusi notebook 5/6 kali ini sempat
> gagal 2x lagi meski TIDAK ada `RandomizedSearchCV`/multiprocessing
> bersarang sama sekali — kode identik yang sebelumnya sukses dalam ~13
> menit tiba-tiba timeout di >25 menit, ternyata karena kontensi CPU
> transien dari aplikasi lain di mesin ini (Discord, Norton, dll.), bukan
> bug kode. Diperbaiki cukup dengan menaikkan `ExecutePreprocessor.timeout`
> dan mengulang — **bukan** masalah nested `n_jobs` seperti sebelumnya.
> Error KEDUA di notebook 6 (`KeyError: 'model'`) adalah bug kode nyata:
> `compare_models()` (`src/evaluation/metrics.py`) meng-set `"model"`
> sebagai index DataFrame (`.set_index("model")`), bukan kolom biasa —
> `df.iloc[0]['model']` gagal, harus pakai `df.index[0]`. Perhatikan pola
> ini di notebook mana pun yang memakai `compare_models()`.

**Pembersihan model kandidat notebook 5 (2026-07-21, lanjutan)** —
⚠️ **DIBATALKAN 2026-08-27** (lihat "Audit menyeluruh notebooks/01–06" di
bawah): pemangkasan ini membuat `notebooks/06_final_evaluation.ipynb` tidak
bisa dijalankan ulang. Sekarang nb05 Bagian 11 menyimpan **semua** kandidat
lagi, pakai `joblib compress=3` sehingga `randomforest_expE_retuned.pkl`
turun dari 1,29GB → ~123MB; nb06 **tidak lagi frozen**. Paragraf di bawah
dipertahankan sebagai riwayat.

Isi keputusan lama (2026-07-21): setelah
`notebooks/06_final_evaluation.ipynb` selesai dan hasilnya terekam
permanen di atas, 8 model kandidat yang tadinya disimpan tanpa syarat
dipangkas jadi **2 saja** untuk reklaim disk (~1,4GB, didominasi
`randomforest_expE_retuned.pkl` yang sendirian 1,29GB):
- **Disimpan**: `xgboost_expF_softbalance.pkl` (kandidat all-around
  terbaik) dan `stacking_expH_meta_logreg.pkl` (accuracy tertinggi
  proyek, 58,2%).
- **Dihapus** (superseded/tidak kompetitif): `randomforest_expE_retuned.pkl`,
  `lightgbm_expC_retuned.pkl` (kalah dari LightGBM baseline resmi di semua
  metrik), `xgboost_expA_default.pkl` & `xgboost_expB_balanced.pkl`
  (keduanya didominasi Exp F), `ensemble_manifest.json`.

**Konsekuensi**: `stacking_expH_meta_logreg.pkl` sekarang **beku**
(frozen) — 3 model dasarnya sudah dihapus, jadi tidak bisa dipakai ulang
langsung tanpa regenerasi (lihat `outputs/models/stacking_manifest.json`
field `"status"`). `notebooks/06_final_evaluation.ipynb` juga jadi
**frozen** senada dengan `notebooks_dds/` — sudah menghasilkan laporan
final sekali, tapi TIDAK bisa dijalankan ulang apa adanya (sel pemuatan
model akan gagal `FileNotFoundError` untuk 4 file yang sudah dihapus).
`notebooks/05_improvement_experiments.ipynb` Bagian 12 sudah diupdate
mengikuti perilaku baru ini (hanya menyimpan 2 model), sehingga eksekusi
ulang notebook 5 di masa depan konsisten dengan keadaan disk saat ini.

**Audit menyeluruh notebooks/01–06 + perbaikan (2026-08-27)**: audit lengkap
6 notebook resmi menemukan beberapa masalah; semua diperbaiki dan notebook
**01→06 dieksekusi ulang penuh** (kernel `bridge-venv` = `.venv`), semua
lolos "Restart & Run All" dengan execution count berurutan tanpa error.

1. **XGBoost "acc-tuned" DIPROMOSIKAN ke default `config.yaml`**
   (keputusan yang sebelumnya "menunggu konfirmasi"). `notebooks/03_modeling.ipynb`
   kini memuat SEMUA hyperparameter dari `configs/config.yaml` via
   `yaml.safe_load` — tidak ada angka di-hardcode di notebook (sesuai
   "Konvensi Kode"). `configs/config.yaml` diupdate: `class_weight: balanced`
   ditambahkan eksplisit ke blok `random_forest` + `lightgbm` (sebelumnya
   hanya default di wrapper `src/models/`). **Hasil test set XGBoost:
   56.1%/0.342/0.532 → 57.5%/0.390/0.554** (naik di accuracy, F1 macro,
   F1 weighted, top-3, top-5 — tanpa trade-off). RF & LightGBM **tidak
   berubah**. LightGBM tetap model utama (F1 macro 0.410 tertinggi).

2. **`notebooks/05_improvement_experiments.ipynb` diperbaiki**: notebook
   sebelumnya **crash** pada "Restart & Run All" — `NameError: best_xgb_proba`
   (Eksperimen G) dan `NameError: best_xgb_key` (Bagian 10), dua nama yang
   tak pernah didefinisikan; execution count tidak berurutan; output
   tersimpan + `nb05_summary.json` + `nb05_test_comparison.csv` berasal dari
   3 versi kode berbeda (termasuk versi `RandomizedSearchCV` lama yang
   melaporkan "Ensemble F1 macro 0.411 = terbaik", kontradiksi dengan
   seluruh narasi proyek). Perbaikan: variabel undefined → `y_proba_a` /
   `xgb_a`; **Eksperimen D (soft-voting 50/50) ditambahkan kembali** ke kode
   (sebelumnya dirujuk nb06 tapi hilang dari sel nb05); Bagian 11 menyimpan
   SEMUA 6 model kandidat + `ensemble_manifest.json` + `stacking_manifest.json`
   (joblib `compress=3`).

3. **`notebooks/06_final_evaluation.ipynb` TIDAK LAGI frozen**: pemangkasan
   model 2026-07-21 membuatnya gagal `FileNotFoundError` di sel ke-3. Karena
   nb05 kini menyimpan semua kandidat lagi (RF Exp E turun 1,29GB → ~123MB
   berkat joblib compress), nb06 kembali bisa dijalankan ulang normal. Load
   model diganti ke `joblib.load` (baca pickle biasa maupun joblib compress).
   `stacking_expH_meta_logreg.pkl` tidak lagi "beku".

4. **`notebooks/04_evaluation.ipynb`**: blok "PERBANDINGAN 164 vs 182 fitur"
   dihapus — blok itu membaca `test_comparison.csv` yang **baru saja ditulis
   notebook itu sendiri**, jadi selalu membandingkan hasil dengan dirinya
   sendiri (output harfiah "+0.00pp" untuk semua). Ablasi DDS yang
   sebenarnya ada di `experiments/2026-07-15/`.

5. **`src/features/dds.py`**: konstanta `DDS_FEATURE_COLUMNS` diperbaiki
   urutannya — dulu blok ns lalu blok ew; sebenarnya selang-seling
   (`ns_dd_S, ew_dd_S, ns_dd_H, …`) seperti yang diproduksi
   `compute_dds_features` dan tersimpan di `data/processed/feature_columns.json`.
   Konstanta hanya dokumentatif (tidak dipakai untuk indexing) tapi
   menyesatkan.

6. **`notebooks/01_data_extraction.ipynb`**: judul plot "Statistik Dataset"
   diperbaiki (dulu "606 File LIN dari BBO", padahal plot menghitung 61.420
   board LIN+PBN **pra-deduplikasi**); "Top-3 kontrak" di ringkasan sekarang
   dari dataset final (dedup) supaya konsisten dengan sel jumlah-per-kelas.
   NB02 provenance heuristik dicek — **sudah benar** (semua file tistis
   berawalan `tistis_`), tidak diubah.

7. **Bersih-bersih**: 19 file output basi dihapus dari `outputs/`
   (`val_comparison.csv` + `*_test.json` tanpa prefix dari run 2026-07-09;
   PNG pra-konsolidasi; model yatim `lightgbm_retuned.pkl` +
   `nb05_xgboost_acc_test.json` dkk. dari run nb05 lama 2026-07-22).

**Hasil test set resmi baru** (`notebooks/04`, konsolidasi + promosi XGBoost):

| Model | Accuracy | Top-3 | Top-5 | F1 Macro | F1 Weighted |
|-------|----------|-------|-------|----------|-------------|
| RandomForest | 46.8% | 77.1% | 87.4% | 0.325 | 0.485 |
| **XGBoost** | **57.5%** | **82.6%** | **90.5%** | 0.390 | 0.554 |
| **LightGBM** | 56.4% | 82.1% | 89.7% | **0.410** | **0.557** |

XGBoost (default config.yaml) kini unggul accuracy/top-3/top-5; LightGBM
(`class_weight="balanced"`) tetap unggul F1 macro + F1 weighted dan **tetap
model utama proyek** (prioritas F1 macro karena class imbalance ekstrem).
`notebooks_dds/` tidak diubah (arsip historis, kini beda dari `notebooks/`
karena promosi XGBoost).

---

## Batas Ruang Lingkup

### Yang ada di dalam ruang lingkup
- Tiga model tree-based: RF, XGBoost, LightGBM
- 164 fitur dari data kartu + lelang
- **Fitur turunan double-dummy solver (`endplay`)** — disetujui masuk
  scope 2026-07-15, setelah ditemukan bahwa akurasi model sudah
  mendekati batas konsistensi bidding manusia sendiri (37.6% pasangan
  open/closed-room BBO sepakat kontrak sama persis). Fitur: DD tricks
  per strain untuk NS/EW (`calc_dd_table`) dan kontrak par
  (`par()`) — dipakai sebagai fitur TAMBAHAN opsional di
  `experiments/`, bukan pengganti 164 fitur kanonik, kecuali hasilnya
  divalidasi cukup berharga untuk dipromosikan ke pipeline utama.
- Target utama: `target_base` (36 kelas: PASS + 35 kontrak)
- Evaluasi: accuracy, F1 macro/weighted, top-k accuracy, SHAP

### Yang di luar ruang lingkup — jangan ditambahkan tanpa diskusi
- Neural network / deep learning
- Reinforcement learning / game theory
- Real-time / streaming prediction
- Web app / API serving
- Target `target` dengan marker doubled/redoubled (66 kelas) sebagai
  primary — gunakan hanya untuk analisis tambahan

---

## Environment

```
Python  : 3.12
Kernel  : "Python 3.12 (Bridge)"  — didaftarkan via ipykernel
Catatan : .venv diblokir Windows Application Control di mesin ini
          Gunakan Python sistem langsung
```

Install dependencies:
```powershell
pip install -e ".[notebook,dev]"
```

Daftarkan Jupyter kernel:
```powershell
python -m ipykernel install --user --name bridge --display-name "Python 3.12 (Bridge)"
```

---

## Struktur Repository

```
src/
  parser/
    lin_parser.py          LINParser, BoardRecord, Hand; tokeniser BBO LIN
    pbn_parser.py          PBNParser; hasil kompatibel dengan BoardRecord/Hand LIN
  features/
    engineer.py            164 fitur kanonik: extract_features(board) → dict
    dds.py                  18 fitur Double-Dummy Solver (endplay): compute_dds_features()
  preprocessing/
    dataset_builder.py     build_dataset(), load_splits() — dukung extra_boards + DDS
  models/
    base.py                BaseModel ABC
    random_forest.py       RFModel (sklearn, class_weight=balanced)
    xgboost_model.py       XGBModel
    lightgbm_model.py      LGBMModel
  evaluation/
    metrics.py             evaluate(), compare_models(), save_results()

notebooks/                    Pipeline RESMI (konsolidasi 2026-07-17; re-run penuh 2026-08-27)
  01_data_extraction.ipynb    Parsing LIN+PBN + DDS → dataset CSV (data/processed/), 182 fitur
  02_eda_features.ipynb       EDA & dokumentasi 182 fitur
  03_modeling.ipynb           Training RF/XGBoost/LightGBM (hyperparameter dari configs/config.yaml)
  04_evaluation.ipynb         Evaluasi final, SHAP, radar
  05_improvement_experiments.ipynb  Eksperimen A–H (retuning, weighting, ensemble, stacking)
  06_final_evaluation.ipynb   Laporan komprehensif 11 kandidat (baseline + Eksperimen A–H)

notebooks_dds/                 ARSIP historis — identik dengan notebooks/ sebelum
                               konsolidasi 2026-07-17. TIDAK dieksekusi ulang; kini
                               berbeda dari notebooks/ (belum punya promosi XGBoost 2026-08-27)

scripts/
  run_pipeline.py             Pipeline lengkap: parse → train → eval — BELUM diupdate
                               ke 182 fitur/PBN, jangan jalankan tanpa update (lihat
                               "Status Proyek")
  report.py                   Generate REPORT.md dari JSON hasil tersimpan
  validate_parser.py          Smoke test parser LIN
  validate_features.py        Validasi output feature engineering

docs/
  SUMMARY.md                   Ringkasan proyek satu halaman — BELUM disinkronkan ke
                                konsolidasi 2026-07-17, masih deskripsikan 164 fitur lama
  ARCHITECTURE.md              Alur pipeline, tanggung jawab modul, keputusan desain
  FEATURES.md                  Data dictionary lengkap 164 fitur kanonik (18 fitur DDS
                                belum didokumentasikan di sini)

configs/config.yaml           Hyperparameter + path (sumber kebenaran)
data/raw/                     606 file .lin (BBO) — tidak di-git
data/raw_pbn/                 1.390 file .pbn (non-BBO) — tidak di-git
data/processed/               CSV split + artefak encoder, 182 fitur/49.755 board — tidak di-git
outputs/models/               Model tersimpan .pkl — tidak di-git
outputs/results/               PNG visualisasi + JSON hasil — tidak di-git
```

---

## Pipeline Data (urutan)

```
1. LINParser.parse_directory("data/raw/")  →  list[BoardRecord]
2. extract_features(board)                 →  dict (164 fitur + metadata + 3 target)
3. build_dataset()                         →  deduplicate, encode label, split 70/15/15
                                               (group-aware: dikelompokkan per papan
                                               fisik _source_file+_board_number, supaya
                                               pasangan open/closed-room BBO — kartu
                                               identik — tidak terpecah lintas split)
4. Model.fit(X_train, y_train)
5. evaluate(y_true, y_pred, y_proba, le)   →  dict metrik
```

Semua parameter (path, hyperparameter, seed) diambil dari `configs/config.yaml`.

---

## Kelompok Fitur (164 total)

| Kelompok | Prefix | Jml | Keterangan |
|----------|--------|-----|------------|
| Per-seat hand | `N_`, `E_`, `S_`, `W_` | 96 | HCP total, HCP per suit, panjang suit, stopper, controls, LTC, balanced, void/singleton/doubleton, longest suit |
| Partnership | `ns_`, `ew_` | 44 | HCP gabungan, LTC, fit per suit, has_fit (≥8), best suit, NT stoppers, both_balanced |
| HCP advantage | `hcp_ns_advantage` | 1 | `ns_hcp − ew_hcp` |
| Deal context | `dealer_`, `vuln_` | 8 | dealer one-hot ×4, vulnerability one-hot ×4 |
| Auction | `auction_`, `opening_` | 15 | panjang lelang, competitive, ns/ew bid count, doubled, opening level/strain, alerts |

Metadata (prefix `_`) tidak digunakan sebagai fitur ML:
`_board_number`, `_source_file`, `_room`, `_declarer`, `_result`, `_tricks_made`

---

## Target Variable

| Kolom | Kelas | Keterangan |
|-------|-------|------------|
| `target_base` | 36 | **Primary.** "PASS" atau "{level}{strain}" misal "3N", "4S" |
| `target` | ≤66 | Termasuk "x"/"xx" untuk doubled/redoubled |
| `target_category` | 5 | Pass / Partscore / Game / SmallSlam / GrandSlam |

---

## Hasil Final

*(test set, `notebooks/04_evaluation.ipynb`, konsolidasi 2026-07-17 +
promosi XGBoost 2026-08-27 — 606 file `.lin` + 1.390 file `.pbn` /
**49.755 papan** / **182 fitur** (164 kanonik + 18 DDS) / 36 kelas, split
**group-aware**. Superseded angka 164-fitur/10.223-papan sebelumnya — lihat
"Status Proyek" di atas)*

| Model | Accuracy | Top-3 | Top-5 | F1 Macro | F1 Weighted |
|-------|----------|-------|-------|----------|-------------|
| RandomForest | 46.8% | 77.1% | 87.4% | 0.325 | 0.485 |
| **XGBoost** | **57.5%** | **82.6%** | **90.5%** | 0.390 | 0.554 |
| **LightGBM** | 56.4% | 82.1% | 89.7% | **0.410** | **0.557** |

LightGBM (`class_weight="balanced"`) unggul di F1 macro/F1 weighted dan
**tetap model utama proyek** (prioritas F1 macro karena class imbalance
ekstrem); XGBoost (default `configs/config.yaml`, dipromosikan 2026-08-27
menggantikan hyperparameter "acc-tuned" usang) unggul di accuracy/top-3/top-5.
Sumber: `outputs/results/test_comparison.csv`. Riwayat lengkap kenaikan
bertahap (52.1% → 56.4% F1w, lewat penambahan DDS + data non-BBO PBN +
`class_weight`) ada di "Status Proyek" di atas dan
`experiments/2026-07-15/README.md`. Perbandingan 11 kandidat (retuning,
ensemble, stacking) di skala 49.755-board ada di
`notebooks/05_improvement_experiments.ipynb` + `06_final_evaluation.ipynb`.

---

## Konvensi Kode

- Semua fitur yang masuk ke model harus numerik/biner — tidak ada kolom string
- Daftar kolom fitur tersimpan di `data/processed/feature_columns.json`;
  gunakan `load_splits()` agar konsisten
- Hyperparameter ada di `configs/config.yaml` — tidak boleh hardcode di
  notebook. `notebooks/03_modeling.ipynb` memuatnya via `yaml.safe_load`
  (`CONFIG`); `configs/config.yaml` menyertakan `class_weight: balanced`
  untuk `random_forest` + `lightgbm`. `notebooks/05` memakai grid manual
  eksplisit (eksperimen tuning — dikecualikan dari aturan ini).
- Model disimpan ke `outputs/models/` dalam format `.pkl` (`joblib` untuk
  kandidat besar nb05; `joblib.load` membaca keduanya)
- Hasil evaluasi disimpan ke `outputs/results/` dalam format `.json` dan `.png`

---

## Invariant — Jangan Ubah Tanpa Alasan Kuat

| Apa | Alasan |
|-----|--------|
| Token handling di `lin_parser.py` | Divalidasi pada 10.223 board (506 file) tanpa error parsing |
| Random seed `42` | Hasil dalam notebook bergantung pada split ini |
| Urutan kolom fitur | Model .pkl terserialisasi dengan urutan ini |
| `target_base` sebagai target utama | Semua model dilatih dan dievaluasi di atas ini |
| Rasio split 70/15/15 | Digunakan di semua perbandingan model |
| Split group-aware (`StratifiedGroupKFold` per `_source_file`+`_board_number`) | Mencegah pasangan open/closed-room BBO (kartu identik) terpecah lintas train/val/test — lihat "Status Proyek" |
