# CLAUDE.md — Bridge Contract Prediction

## Ringkasan Proyek
Penelitian skripsi membandingkan tiga algoritma ensemble berbasis pohon
(Random Forest, XGBoost, LightGBM) untuk memprediksi kontrak optimal
Contract Bridge dari rekaman BBO LIN format.

## Status Proyek (September 2026)
**Perubahan arah penelitian (2026-09-18)**: target diganti dari klasifikasi
36-kelas (`target_base`) menjadi **klasifikasi biner `matches_par_contract`** — apakah
kontrak yang benar-benar dibid (auction) sama level+strain dengan kontrak
**par** menurut Double-Dummy Solver. `notebooks/01-06` adalah pipeline
resmi saat ini: 606 file `.lin` + 1.390 file `.pbn` gabungan, **174 fitur**
model (164 kanonik + 10 trik DD per strain; 8 kolom par DDS dikeluarkan
dari fitur karena jadi bahan langsung label — lihat leakage di bawah),
**49.755 board**, target biner (25.2% optimal / 74.8% tidak optimal).
**Dieksekusi ulang penuh 01→06 pada 2026-09-18** — semua notebook lolos
"Restart & Run All" berurutan tanpa error. Detail lengkap di bawah
("Perubahan arah penelitian — target biner `matches_par_contract` (2026-09-18)").
**Catatan**: `scripts/run_pipeline.py` **belum diupdate** dan masih
membangun pipeline 164-fitur/LIN-only/36-kelas yang lama — lihat peringatan
di bawah, jangan dijalankan tanpa diupdate dulu. `docs/SUMMARY.md` juga
masih mendeskripsikan pipeline lama — belum disinkronkan. `scripts/report.py`
juga belum diupdate untuk metrik biner (masih mereferensikan
`top_3_accuracy` yang sudah tidak ada di `nb04_summary.json`/`nb06_summary.json`
— tidak crash, hanya menampilkan "n/a" untuk kolom itu).

Riwayat lengkap era 36-kelas (Juli–Agustus 2026: perbaikan split, penambahan
fitur DDS, perluasan data PBN, konsolidasi notebook, promosi XGBoost, audit
menyeluruh, dll.) dipertahankan di bawah sebagai riwayat — semua angka di
paragraf-paragraf itu **superseded** oleh perubahan arah 2026-09-18 di atas,
tapi keputusan data/parser/fitur-nya (606 LIN, 1.390 PBN, 164 fitur kanonik,
18 fitur DDS mentah) tetap jadi fondasi pipeline saat ini.

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
komprehensif (2026-07-21)** — ⚠️ **Eksperimen D, G, H (ensemble + stacking)
DIHAPUS 2026-08-31** (lihat "Audit menyeluruh" di bawah): di luar ruang
lingkup penelitian (RF vs XGBoost vs LightGBM sebagai model tunggal).
`notebooks/05` sekarang hanya Eksperimen **A, B, C, E, F**; `notebooks/06`
mengevaluasi **8 kandidat** (3 baseline + 5 retuning). Paragraf di bawah
dipertahankan sebagai riwayat.

Isi lama (2026-07-21): `notebooks/05_improvement_experiments.ipynb`
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
- **Eksperimen G** *(DIHAPUS 2026-08-31 — di luar scope)* — bobot ensemble
  XGBoost/LightGBM dicari (bukan 50/50 tetap): `w_xgb=0.7` sedikit lebih baik
  dari 50/50 di F1 macro (0.397 val vs 0.393), tapi tidak mengalahkan model
  tunggal terbaik manapun.
- **Eksperimen H** *(DIHAPUS 2026-08-31 — di luar scope)* — stacking
  (`LogisticRegression` di atas `predict_proba` RF+XGBoost+LightGBM):
  akurasi tertinggi dari seluruh kandidat proyek saat itu (58.2% test),
  tapi F1 macro-nya (0.384) TERENDAH di antara kandidat non-RandomForest —
  stacking condong ke kelas mayoritas.
- *(Eksperimen D — ensemble soft-voting 50/50 — juga dihapus 2026-08-31.)*

**Hasil akhir komprehensif (test set, `notebooks/06_final_evaluation.ipynb`)** —
diurutkan F1 macro. *Angka setelah promosi XGBoost (2026-08-27) + penghapusan
ensemble/stacking (2026-08-31): 8 kandidat.*

| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| **LightGBM (baseline resmi)** | 56.4% | **0.410** | 0.557 |
| XGBoost (Exp B: balanced penuh) | 52.4% | 0.407 | 0.539 |
| XGBoost (Exp F: soft-balanced α=0.25) | **57.5%** | 0.406 | **0.560** |
| LightGBM (Exp C: retuned) | 55.6% | 0.394 | 0.544 |
| RandomForest (Exp E: retuned) | 48.9% | 0.390 | 0.507 |
| XGBoost (Exp A: retuned) | 57.5% | 0.390 | 0.554 |
| XGBoost (baseline resmi) | 57.5% | 0.390 | 0.554 |
| RandomForest (baseline resmi) | 46.8% | 0.325 | 0.485 |

*(Kandidat historis yang dihapus, untuk rujukan: Stacking H 58.2%/0.384/0.559 ·
Ensemble D 50/50 56.9%/0.400/0.555 · Ensemble G 57.3%/0.397/0.556.)*

> Tabel di atas: angka **setelah** promosi XGBoost default config.yaml
> (2026-08-27, lihat "Audit menyeluruh" di bawah). Baris "XGBoost (baseline
> resmi)" dulu 56.1%/0.342/0.532 ("acc-tuned lama") — sekarang identik dengan
> Exp A karena keduanya memakai default config.yaml. Sisa tabel tidak berubah
> (deterministik, model & data sama).

**Kesimpulan akhir**: **LightGBM (baseline resmi, tidak berubah) tetap
model utama proyek** — F1 macro tertinggi (prioritas utama karena class
imbalance ekstrem), sesuai semua analisis. **XGBoost Exp F (soft-balanced
α=0.25)** adalah alternatif "all-around terbaik": F1 macro nyaris identik
dengan LightGBM (selisih ~0.4pp, dalam rentang noise) namun unggul di
accuracy (+1.1pp), F1 weighted, dan top-3/top-5 — sepadan dipertimbangkan
sebagai alternatif resmi jika prioritas condong ke accuracy/F1-weighted.
**Update 2026-08-27**: `outputs/models/xgboost.pkl` resmi kini memakai
default config.yaml (promosi dari "acc-tuned"); `lightgbm.pkl` &
`randomforest.pkl` tidak berubah. Exp F tetap rekomendasi (alternatif),
bukan promosi. **Update 2026-08-31**: Eksperimen D/G/H (ensemble + stacking)
dihapus dari notebook — di luar ruang lingkup. Detail lengkap, grafik
(confusion matrix, feature importance, SHAP, radar chart), dan tabel mentah
ada di `notebooks/06_final_evaluation.ipynb` dan
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

**Penghapusan ensemble & stacking dari notebook (2026-08-31)**: atas
permintaan, Eksperimen **D** (soft-voting 50/50), **G** (soft-voting bobot
dicari), dan **H** (stacking `LogisticRegression`) dihapus dari
`notebooks/05` dan `notebooks/06` — di luar ruang lingkup penelitian, yang
membandingkan RF vs XGBoost vs LightGBM sebagai **model tunggal**. Poin 2 & 3
di atas sebagian tergantikan: `notebooks/05` sekarang hanya Eksperimen
A/B/C/E/F dan menyimpan 5 model kandidat (tanpa `ensemble_manifest.json` /
`stacking_manifest.json` / `stacking_expH_meta_logreg.pkl` — file-file itu
dihapus); `notebooks/06` mengevaluasi 8 kandidat (3 baseline + 5 retuning),
tidak lagi 11. Keduanya dijalankan ulang; angka model tunggal tidak berubah
(deterministik). Kesimpulan tidak berubah: LightGBM model utama, XGBoost
Exp F alternatif all-around.

**Eksperimen seleksi fitur SHAP (2026-08-31)**: diminta menguji seleksi
fitur berbasis SHAP. `experiments/2026-08-31/01_shap_feature_selection.ipynb` —
ranking `mean |SHAP|` (LightGBM & XGBoost, sub-sampel train 2.000) → retrain
di top-k → bandingkan ke baseline 182 fitur. **Hasil**: top-40 fitur SHAP
(LightGBM) **setara** baseline 182 di test set (F1 macro 0.4142 vs 0.4101 —
selisih di dalam noise urutan kolom ~±0.8pp), pakai 78% fitur lebih sedikit.
Kurva datar mulai k≈40. **Redundansi per-seat besar** (top-40 simpan cuma
5/96 fitur per-kursi, 0/8 dealer/vuln). **Sirkularitas auction terkonfirmasi
kuantitatif**: 3 fitur SHAP teratas = `auction_ew_bids`/`auction_len`/
`auction_ns_bids` (dihitung dari lelang yang sudah selesai). **Rekomendasi:
TIDAK dipromosikan** ke `notebooks/` (gain di dalam noise, ongkos mengubah
`feature_columns.json` invariant tidak sepadan) — berharga sebagai analisis
fitur untuk skripsi. Tindak lanjut yang disarankan: ablasi eksplisit fitur
auction sirkular. `notebooks/` + `data/processed/` + `outputs/` resmi TIDAK
tersentuh. Detail: `experiments/2026-08-31/README.md`.

**Perubahan arah penelitian — target biner `matches_par_contract` (2026-09-18)**: atas
permintaan eksplisit, seluruh pipeline resmi diganti dari klasifikasi
36-kelas (`target_base`) menjadi **klasifikasi biner**: apakah kontrak yang
benar-benar dibid manusia optimal (sama persis level+strain dengan kontrak
par DDS) atau tidak.

**Definisi label `matches_par_contract`** (disepakati lewat diskusi eksplisit sebelum
implementasi): `matches_par_contract = 1` jika level+strain `target_base` sama persis
dengan level+strain kontrak par DDS (`dd_par_level` + `dd_par_denom`), untuk
SEMUA board apa pun sisi declarer-nya. Declarer dan skor tidak
diperhitungkan, hanya level+strain. Catatan penting yang mengoreksi asumsi
awal: `endplay.dds.par()` cuma menghasilkan **satu** kontrak par per board
(bukan par terpisah untuk NS dan EW), jadi tidak ada konsep "par dari sisi
NS" yang berbeda dari "par dari sisi EW" — pertanyaan "fokus ke NS" dari
diskusi awal diselesaikan dengan membandingkan SEMUA board tanpa filter sisi
declarer. Dihitung di `build_dataset()` (`src/preprocessing/dataset_builder.py`),
BUKAN di `src/features/dds.py` (yang tidak diubah sama sekali) — dari
kolom `dd_par_level`/`dd_par_denom_*` yang sudah di-merge, dibandingkan
dengan `target_base`.

**Anti-leakage**: 8 kolom DDS yang jadi bahan langsung label
(`dd_par_level`, `dd_par_denom_S/H/D/C/N`, `dd_par_score`,
`dd_par_declarer_is_ns`) dikeluarkan permanen dari `feature_columns.json`
(`_LEAKAGE_COLS` di `dataset_builder.py`) — tetap tersimpan di CSV untuk
analisis/audit (lihat korelasi di `notebooks/02_eda_features.ipynb` bagian
7, yang menunjukkan `dd_par_level`/`dd_par_declarer_is_ns` berkorelasi jauh
lebih kuat dengan `matches_par_contract` dibanding fitur lain manapun — persis seperti
diharapkan, karena keduanya membentuk label). 10 kolom trik DD per strain
(`ns_dd_*`/`ew_dd_*`) TIDAK otomatis dikeluarkan — keputusan pakai/buang
divalidasi empiris lewat ablasi baru di `notebooks/03_modeling.ipynb`
bagian 1b (RandomForest cepat, 174 fitur vs 164 fitur kanonik saja,
dibandingkan di validation set, ambang keputusan +0.5pp F1 macro). **Hasil
ablasi**: 174 fitur (F1 macro val 0.7348) mengalahkan 164 fitur (0.6927)
sebesar **+4.21pp** — jauh di atas ambang, trik DD per strain
**dipertahankan**. `feature_columns.json` resmi = **174 fitur** (164
kanonik + 10 trik DD).

**Restrukturisasi `build_dataset()`** (`src/preprocessing/dataset_builder.py`):
blok merge fitur DDS dipindah dari tahap akhir ("3b", setelah label
encoding) ke tahap awal (setelah ekstraksi fitur, sebelum cleaning/encoding)
— perlu karena `matches_par_contract` sekarang bergantung pada kolom DDS, jadi harus
tersedia sebelum label di-encode. Efek samping kecil yang disengaja: dedup
baris sekarang mempertimbangkan kolom DDS juga (sebelumnya tidak) — risiko
perubahan hasil dedup minimal karena DDS deterministik dari tangan yang
sudah tercakup fitur kanonik.

**Metrik evaluasi diganti untuk biner** (`src/evaluation/metrics.py`):
top-k accuracy (tidak bermakna untuk 2 kelas) dihapus, diganti ROC-AUC,
PR-AUC (average precision), dan precision/recall/F1 kelas positif — di
samping accuracy/F1 macro/F1 weighted yang tetap ada. `configs/config.yaml`
(`evaluation.top_k`) juga dihapus, `evaluation.metrics` ditambah `roc_auc`/
`average_precision`. `src/models/xgboost_model.py`: `eval_metric="mlogloss"`
(multi-class-only, akan salah untuk 2 kelas) diganti `"logloss"` — begitu
juga instansiasi `XGBClassifier` langsung di sel learning-curve
`notebooks/03` dan grid manual `notebooks/05`.

**Insiden lingkungan (ditemukan & diperbaiki di sesi yang sama)**: instalasi
Python 3.12 dasar yang jadi basis `.venv` proyek (`AppData/Local/Programs/
Python/Python312`) hilang dari mesin ini (kemungkinan terhapus di luar
sesi) — `.venv\Scripts\python.exe` gagal total ("No Python at ..."), dan
tidak ada Python/conda lain yang terpasang sama sekali. Diperbaiki dengan
`py install 3.12` (Python launcher, mengunduh installer resmi dari
python.org) lalu `.venv` dibuat ulang dari nol + `pip install -e
".[notebook,dev]"` + registrasi ulang kernel Jupyter (`python -m ipykernel
install --user --name bridge --display-name "Python 3.12 (Bridge)"`).
Ditemukan sekaligus 2 bug terpisah di `pyproject.toml`: (1)
`build-backend = "setuptools.backends.legacy:build"` tidak valid (modul
tidak ada), menyebabkan SEMUA instalasi editable gagal — diperbaiki ke
`"setuptools.build_meta"` (nilai standar); (2) `endplay` (DDS) sebelumnya
cuma ada di extra opsional `experiments`, padahal sekarang **wajib** untuk
pipeline resmi (label `matches_par_contract` butuh DDS) — dipindah ke dependencies
dasar, `experiments` extra sekarang cuma `imbalanced-learn`.

**Hasil (test set, semua notebook 01→06 dieksekusi ulang penuh 2026-09-18,
tanpa error)**:

| Model | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Baseline resmi RandomForest | 80.8% | 0.736 | 0.805 | 0.800 | 0.634 |
| Baseline resmi XGBoost | 81.0% | 0.691 | 0.786 | 0.800 | 0.634 |
| Baseline resmi LightGBM | 79.9% | 0.741 | 0.802 | 0.814 | 0.658 |
| Exp A — XGBoost retuned (= default config.yaml) | 82.5% | 0.731 | 0.810 | 0.820 | 0.670 |
| Exp B — XGBoost + sample_weight balanced penuh | 81.0% | 0.749 | 0.810 | 0.818 | 0.667 |
| Exp C — LightGBM retuned | 81.0% | 0.749 | 0.810 | 0.820 | 0.670 |
| Exp E — RandomForest retuned (`max_features` dkk.) | 81.2% | 0.745 | 0.810 | 0.818 | 0.666 |
| **Exp F — XGBoost soft-balanced (α=0.25)** | **82.3%** | **0.750** | **0.817** | 0.818 | 0.663 |

Baseline mayoritas (selalu tebak "tidak optimal") = 74.8% accuracy — semua
model mengalahkannya, dan F1 macro (bukan cuma accuracy) menunjukkan model
benar-benar belajar membedakan kedua kelas, bukan cuma menebak mayoritas.
Tidak ada tanda leakage residual (ROC-AUC ~0.80-0.82, bukan mendekati 1.0).

**Kesimpulan**: kandidat **terbaik keseluruhan (F1 macro DAN accuracy
sama-sama kompetitif) adalah XGBoost Eksperimen F (soft-balanced,
`sample_weight` parsial α=0.25)** — F1 macro tertinggi (0.7498) sekaligus
accuracy tertinggi kedua (82.3%, cuma 0.2pp di bawah Exp A). Ini beda dari
era 36-kelas (di mana LightGBM class_weight vs XGBoost accuracy-tinggi
adalah trade-off yang jelas) — untuk `matches_par_contract`, satu kandidat XGBoost
unggul di kedua sumbu sekaligus. **Baseline resmi TIDAK diubah**
(`outputs/models/{randomforest,xgboost,lightgbm}.pkl` masih hyperparameter
`configs/config.yaml` apa adanya) — keputusan promosi kandidat Eksperimen F
ke baseline resmi belum diambil, konsisten dengan konvensi proyek
("`notebooks/05` tidak menimpa baseline resmi", lihat `notebooks/06`).
Detail lengkap, grafik, dan tabel mentah ada di
`notebooks/03_modeling.ipynb` (ablasi fitur), `notebooks/04_evaluation.ipynb`,
`notebooks/05_improvement_experiments.ipynb`
(`outputs/results/nb05_summary.json`/`nb05_test_comparison.csv`), dan
`notebooks/06_final_evaluation.ipynb`
(`outputs/results/nb06_final_comparison.csv`/`nb06_summary.json`).

`notebooks_dds/` dan `experiments/` **tidak disentuh** — tetap arsip
historis skema 36-kelas lama.

**Perapian nama kolom fitur (2026-09-20)**: diminta merapikan penamaan
dataset. Ditemukan dua ketidakkonsistenan (bukan bug, murni kosmetik):
(1) `hcp_ns_advantage` tidak mengikuti konvensi prefix `ns_`/`ew_` yang
dipakai semua kolom partnership lain → diganti **`ns_hcp_advantage`**;
(2) `{seat}_longest_{S,H,D,C}` (one-hot "suit ini terpanjang?") gampang
tertukar dengan `{seat}_len_{S,H,D,C}` (panjang suit, nilai integer) →
diganti **`{seat}_is_longest_{S,H,D,C}`**. Diperbaiki di
`src/features/engineer.py`; kolom `label` (LabelEncoder atas `target_col`)
dicek dan **tidak diubah** — sempat terlihat seperti duplikat `matches_par_contract`
(nilainya memang identik sekarang karena `target_col` sudah biner), tapi
ternyata dipakai langsung sebagai `y_train`/`y_test` di 5 notebook (02–06)
+ `scripts/run_pipeline.py` + internal `dataset_builder.py` — bukan sisa,
sengaja jadi abstraksi target yang seragam terlepas dari `target_col` aktif.
Karena rename menyentuh kolom di dataset resmi, `notebooks/01→06`
**dieksekusi ulang penuh** — cache DDS (`data/*_dds_cache*.csv`) dipakai
ulang (tidak disentuh oleh rename, hanya nama fitur kanonik yang berubah)
sehingga rebuild jauh lebih cepat dari komputasi DDS awal. Semua 6 notebook
lolos tanpa error; hasil test set **identik** dengan angka yang sudah
tercatat di atas (XGBoost Exp F tetap 82.3%/F1 macro 0.750, dst.) —
mengonfirmasi rename murni kosmetik, tidak mengubah data atau model.
`docs/FEATURES.md` diupdate mengikuti nama baru.

**Rename kolom target: `is_optimal` → `matches_par_contract` (2026-09-20,
lanjutan)**: atas permintaan eksplisit, kolom target biner diganti nama dari
`is_optimal` menjadi **`matches_par_contract`** — nama baru menggambarkan
definisi persis (kecocokan level+strain dengan kontrak par DDS) tanpa istilah
"optimal" yang lebih longgar/ambigu. **Definisi, nilai, dan hasil model tidak
berubah** — murni rename. Diperbaiki di `src/preprocessing/dataset_builder.py`
(`_TARGET_COLS`, kolom `df["matches_par_contract"]`, docstring `target_col`),
`src/evaluation/metrics.py` (docstring), `configs/config.yaml` (komentar),
`notebooks/01-06` (semua sel kode & markdown yang mereferensikan nama kolom,
termasuk kunci JSON `nb04_summary.json`: `is_optimal_rate_test` →
`matches_par_contract_rate_test` — tidak ada notebook lain yang membaca kunci
itu, aman diganti), dan `docs/SUMMARY.md`. Notebook 01→06 **dieksekusi ulang
penuh** lagi setelah rename (cache DDS dipakai ulang); hasil test set
identik dengan sebelumnya. `notebooks_dds/` dan `experiments/` **tidak
disentuh** (arsip historis skema 36-kelas lama, tidak pernah punya kolom
ini). Semua paragraf riwayat "Status Proyek" di atas (termasuk yang
bertanggal sebelum 2026-09-20) sudah ditimpa memakai nama baru
`matches_par_contract` demi konsistensi dokumen — kolom itu sebenarnya baru
disebut `matches_par_contract` mulai hari ini, dokumen ini cuma
menyamakan penamaan secara retroaktif; tidak ada perubahan kode/data yang
tersirat pada tanggal-tanggal sebelum 2026-09-20 akibat ini.

---

## Batas Ruang Lingkup

### Yang ada di dalam ruang lingkup
- Tiga model tree-based: RF, XGBoost, LightGBM
- 164 fitur kanonik dari data kartu + lelang
- **Fitur turunan double-dummy solver (`endplay`)** — disetujui masuk
  scope 2026-07-15. 10 fitur trik DD per strain (`ns_dd_*`/`ew_dd_*`)
  jadi fitur model permanen (divalidasi lewat ablasi, `notebooks/03`);
  8 kolom par DDS (`dd_par_*`) dikeluarkan dari fitur model karena jadi
  bahan langsung label `matches_par_contract` (leakage) — lihat "Status Proyek"
  2026-09-18.
- **Target utama: `matches_par_contract` (biner)** — apakah kontrak yang dibid
  manusia sama level+strain dengan kontrak par DDS. Menggantikan
  `target_base` (36 kelas) sejak 2026-09-18; `target_base`/`target`/
  `target_category` tetap tersimpan di CSV sebagai kolom historis/analisis,
  bukan lagi target training.
- Evaluasi: accuracy, F1 macro/weighted/positif, ROC-AUC, PR-AUC, SHAP

### Yang di luar ruang lingkup — jangan ditambahkan tanpa diskusi
- Neural network / deep learning
- Reinforcement learning / game theory
- Real-time / streaming prediction
- Web app / API serving
- Target `target_base` (36 kelas) atau `target` dengan marker
  doubled/redoubled (66 kelas) sebagai primary — status 2026-09-18: hanya
  untuk analisis tambahan/historis, `matches_par_contract` biner adalah target utama
- **Ensemble / soft-voting / stacking / meta-learning** dari ketiga model —
  dicoba di notebook 5 (Eksperimen D/G/H) lalu **dihapus 2026-08-31**;
  penelitian membandingkan RF vs XGBoost vs LightGBM sebagai model tunggal

---

## Environment

```
Python  : 3.12
Kernel  : "bridge" / "Python 3.12 (Bridge)"  — didaftarkan via ipykernel
Catatan : .venv sempat rusak total 2026-09-18 (install Python dasarnya
          hilang dari mesin) — diperbaiki dengan `py install 3.12` +
          rebuild .venv dari nol. Lihat "Status Proyek" 2026-09-18 untuk
          detail insiden (termasuk bug build-backend di pyproject.toml).
```

Install dependencies (`endplay`/DDS sekarang wajib, bukan opsional — target
`matches_par_contract` butuh DDS):
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
                            (tidak diubah oleh perubahan target biner — matches_par_contract
                            dihitung di dataset_builder.py dari kolom ini)
  preprocessing/
    dataset_builder.py     build_dataset(), load_splits() — dukung extra_boards + DDS;
                            menghitung target biner matches_par_contract (par DDS vs target_base)
                            dan mengeluarkan 8 kolom par DDS dari feature_columns.json
                            (leakage) — lihat "Status Proyek" 2026-09-18
  models/
    base.py                BaseModel ABC
    random_forest.py       RFModel (sklearn, class_weight=balanced)
    xgboost_model.py       XGBModel
    lightgbm_model.py      LGBMModel
  evaluation/
    metrics.py             evaluate(), compare_models(), save_results()

notebooks/                    Pipeline RESMI (konsolidasi 2026-07-17; target biner
                               matches_par_contract sejak 2026-09-18; re-run penuh 2026-09-18)
  01_data_extraction.ipynb    Parsing LIN+PBN + DDS → dataset CSV (data/processed/),
                               target_col='matches_par_contract'
  02_eda_features.ipynb       EDA matches_par_contract & dokumentasi 174 fitur model
  03_modeling.ipynb           Ablasi fitur trik DD (1b) + training RF/XGBoost/LightGBM
                               (hyperparameter dari configs/config.yaml)
  04_evaluation.ipynb         Evaluasi final (matches_par_contract), SHAP, radar
  05_improvement_experiments.ipynb  Eksperimen A/B/C/E/F (retuning hyperparameter + class weighting model tunggal)
  06_final_evaluation.ipynb   Laporan komprehensif 8 kandidat (3 baseline + 5 retuning A/B/C/E/F)

notebooks_dds/                 ARSIP historis — skema 36-kelas lama (sebelum konsolidasi
                               2026-07-17 dan sebelum perubahan target biner 2026-09-18).
                               TIDAK dieksekusi ulang.

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
data/processed/               CSV split + artefak encoder, 174 fitur model/49.755 board,
                               target matches_par_contract (biner) — tidak di-git
outputs/models/               Model tersimpan .pkl — tidak di-git
outputs/results/               PNG visualisasi + JSON hasil — tidak di-git
```

---

## Pipeline Data (urutan)

```
1. LINParser.parse_directory("data/raw/")  →  list[BoardRecord]
2. extract_features(board)                 →  dict (164 fitur + metadata + 3 target lama)
2b. compute_dds_features() + merge         →  +18 kolom DDS; matches_par_contract dihitung dari
                                               dd_par_level/dd_par_denom vs target_base
3. build_dataset()                         →  deduplicate, encode label (matches_par_contract),
                                               split 70/15/15 (group-aware: dikelompokkan
                                               per papan fisik _source_file+_board_number,
                                               supaya pasangan open/closed-room BBO —
                                               kartu identik — tidak terpecah lintas split);
                                               8 kolom par DDS dikeluarkan dari
                                               feature_columns.json (leakage)
4. Model.fit(X_train, y_train)
5. evaluate(y_true, y_pred, y_proba, le)   →  dict metrik (accuracy, F1, ROC-AUC, PR-AUC)
```

Semua parameter (path, hyperparameter, seed) diambil dari `configs/config.yaml`.

---

## Kelompok Fitur (174 total, model resmi sejak 2026-09-18)

| Kelompok | Prefix | Jml | Keterangan |
|----------|--------|-----|------------|
| Per-seat hand | `N_`, `E_`, `S_`, `W_` | 96 | HCP total, HCP per suit, panjang suit, stopper, controls, LTC, balanced, void/singleton/doubleton, longest suit |
| Partnership | `ns_`, `ew_` | 44 | HCP gabungan, LTC, fit per suit, has_fit (≥8), best suit, NT stoppers, both_balanced |
| HCP advantage | `ns_hcp_advantage` | 1 | `ns_hcp − ew_hcp` |
| Deal context | `dealer_`, `vuln_` | 8 | dealer one-hot ×4, vulnerability one-hot ×4 |
| Auction | `auction_`, `opening_` | 15 | panjang lelang, competitive, ns/ew bid count, doubled, opening level/strain, alerts |
| Double-Dummy (trik) | `ns_dd_*`, `ew_dd_*` | 10 | trik double-dummy per strain (S/H/D/C/N) untuk NS & EW — divalidasi lewat ablasi (`notebooks/03`), naik F1 macro val +4.21pp vs tanpanya |

164 fitur di atas (tanpa baris DDS) = **164 fitur kanonik**. Total dengan
trik DD = **174 fitur model**.

**Dikeluarkan dari fitur model (leakage, bukan karena tidak berguna)**: 8
kolom par DDS — `dd_par_level`, `dd_par_denom_S/H/D/C/N`, `dd_par_score`,
`dd_par_declarer_is_ns` — karena jadi bahan langsung pembentuk label
`matches_par_contract`. Tetap tersimpan di `data/processed/*.csv` untuk analisis,
dikeluarkan dari `feature_columns.json`.

Metadata (prefix `_`) tidak digunakan sebagai fitur ML:
`_board_number`, `_source_file`, `_room`, `_declarer`, `_result`, `_tricks_made`

---

## Target Variable

| Kolom | Kelas | Keterangan |
|-------|-------|------------|
| `matches_par_contract` | 2 | **Primary (sejak 2026-09-18).** 1 jika level+strain `target_base` == level+strain kontrak par DDS, 0 jika tidak. Dihitung di `build_dataset()`. |
| `target_base` | 36 | Historis (target utama sebelum 2026-09-18). "PASS" atau "{level}{strain}" misal "3N", "4S". Tetap tersimpan di CSV, dipakai untuk menghitung `matches_par_contract`. |
| `target` | ≤66 | Historis. Termasuk "x"/"xx" untuk doubled/redoubled |
| `target_category` | 5 | Historis. Pass / Partscore / Game / SmallSlam / GrandSlam |

---

## Hasil Final

*(test set, `notebooks/04_evaluation.ipynb`, target biner `matches_par_contract` sejak
2026-09-18 — 606 file `.lin` + 1.390 file `.pbn` / **49.755 papan** /
**174 fitur** (164 kanonik + 10 trik DD) / 2 kelas, split **group-aware**.
Superseded angka 36-kelas sebelumnya — lihat "Status Proyek" di atas)*

**Baseline resmi** (hyperparameter `configs/config.yaml` apa adanya):

| Model | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | PR-AUC |
|-------|----------|----------|-------------|---------|--------|
| RandomForest | 80.8% | 0.736 | 0.805 | 0.800 | 0.634 |
| XGBoost | 81.0% | 0.691 | 0.786 | 0.800 | 0.634 |
| **LightGBM** | 79.9% | **0.741** | 0.802 | **0.814** | 0.658 |

Baseline mayoritas (selalu tebak "tidak optimal", 74.8% dari test set) =
74.8% accuracy — semua model mengungguli, dan F1 macro menegaskan mereka
membedakan kedua kelas secara nyata, bukan sekadar menebak mayoritas.

**Kandidat terbaik notebook 05/06 (retuning, belum dipromosikan ke
baseline resmi)**: **XGBoost Eksperimen F (soft-balanced, `sample_weight`
parsial α=0.25)** — Accuracy 82.3%, F1 Macro **0.750**, F1 Weighted 0.817,
ROC-AUC 0.818 — unggul di F1 macro DAN nyaris tertinggi di accuracy (0.2pp
di bawah Exp A) sekaligus, beda dari era 36-kelas yang trade-off-nya lebih
tajam. Perbandingan lengkap 8 kandidat (3 baseline + 5 retuning A/B/C/E/F)
ada di `outputs/results/nb06_final_comparison.csv`.

Sumber: `outputs/results/test_comparison.csv` (3 baseline),
`nb06_final_comparison.csv` (8 kandidat). Definisi label, keputusan
anti-leakage, dan hasil ablasi fitur trik-DD ada di "Status Proyek" di atas.

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
| `matches_par_contract` sebagai target utama (sejak 2026-09-18) | Semua model dilatih dan dievaluasi di atas ini; `target_base` tetap perlu ada di CSV karena `matches_par_contract` dihitung darinya |
| 8 kolom par DDS dikeluarkan dari `feature_columns.json` | Kolom itu jadi bahan langsung label `matches_par_contract` — memasukkannya sebagai fitur = leakage |
| Rasio split 70/15/15 | Digunakan di semua perbandingan model |
| Split group-aware (`StratifiedGroupKFold` per `_source_file`+`_board_number`) | Mencegah pasangan open/closed-room BBO (kartu identik) terpecah lintas train/val/test — lihat "Status Proyek" |
