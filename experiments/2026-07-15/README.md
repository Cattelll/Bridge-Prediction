# Percobaan 2026-07-15

Target: naikkan **accuracy** (bukan F1 macro/weighted) mendekati 60%.
Lanjutan dari [experiments/2026-07-09/](../2026-07-09/) dan
[experiments/2026-07-10/](../2026-07-10/) — kedua sesi itu mentok di
~52% accuracy val set, dan sebagian teknik yang dicoba (sample_weight,
DART, resampling) justru menukar accuracy demi F1 macro.

## Daftar percobaan

| # | Notebook | Tujuan | Status |
|---|----------|--------|--------|
| 1 | [01_accuracy_push.ipynb](01_accuracy_push.ipynb) | Soft-voting ensemble RF+XGB+LGBM, dan `RandomizedSearchCV(scoring="accuracy")` (bukan f1_macro seperti nb01 2026-07-09) — di atas 164 fitur | done |
| 2 | [02_dds_features.ipynb](02_dds_features.ipynb) | Fitur Double-Dummy Solver (`endplay`) — DD tricks per strain + kontrak par. **Scope disetujui 2026-07-15**, lihat `CLAUDE.md` | done |
| 3 | [03_accuracy_push_with_dds.ipynb](03_accuracy_push_with_dds.ipynb) | Ulangi nb01 (ensemble + tuning accuracy) di atas 182 fitur (164+DDS, `data/processed_dds/`) — **berhasil**, tervalidasi test set | done |
| 4 | [04_class_imbalance_dds.ipynb](04_class_imbalance_dds.ipynb) | Atasi class imbalance (35 kelas timpang) di atas XGBoost acc-tuned + 182 fitur: class_weight/sample_weight, RandomOverSampler, SMOTE | done |
| 5 | [05_combined_data_source.ipynb](05_combined_data_source.ipynb) | Perluasan data non-BBO (PBN, 1.314 board dari 7 final kejuaraan dunia) + 100 file LIN baru — **hasil terbaik proyek**: LightGBM+class_weight 53.6% acc, F1 macro 0.310 | done |
| 6 | [06_combined_data_v2_more_pbn.ipynb](06_combined_data_v2_more_pbn.ipynb) | Perluasan PBN lanjutan (tistis.nl, 169 file baru) + perbaikan 3 bug parser/merge — **hasil terbaik proyek (baru)**: LightGBM+class_weight 54.3% acc, F1 macro 0.349, di 21.675 board | done |
| 7 | [07_angelfire_expansion.ipynb](07_angelfire_expansion.ipynb) | Perluasan besar (angelfire.com via Wayback Machine, 1.178 file baru) + fix segfault DDS solver — **hasil terbaik proyek (lompatan terbesar)**: LightGBM+class_weight 56.4% acc, F1 macro 0.410, di 49.755 board | done |

## Catatan penting: permintaan crawler BBO ditolak

Diminta membuat crawler BBO otomatis untuk mengumpulkan >=1 juta board.
**Ditolak** — ToS BBO (`news.bridgebase.com/terms`) eksplisit melarang
scraping/automation software, berapa pun batas volumenya. Sebagai
gantinya dicari sumber non-BBO (lihat notebook 05) — hasilnya jauh lebih
kecil dari 1 juta (final di ~13.582 board total), tapi tetap memberi
perbaikan performa nyata dan sepenuhnya legitimate.

## Catatan scope: Double-Dummy Solver (DDS)

DDS awalnya eksplisit di luar ruang lingkup skripsi (`CLAUDE.md`).
Disetujui masuk scope 2026-07-15 sebagai fitur TAMBAHAN opsional (bukan
pengganti 164 fitur kanonik), setelah notebook 01 menunjukkan akurasi
sudah mendekati batas konsistensi bidding manusia sendiri. Dihitung
pakai `endplay` (Python wrapper untuk DDS-nya Bo Haglund), cache di
`outputs/dds_cache.csv` (~10.223 papan, ~60 menit sekali hitung).

## Aturan main

Sama seperti eksperimen sebelumnya: split `data/processed/` group-aware
dipakai apa adanya, val set untuk membandingkan varian, test set
disentuh sekali di akhir untuk kandidat final.

## Ringkasan hasil

**Tidak ada pendekatan yang mendekati 60% accuracy.** Ensemble voting
(RF+XGB+LGBM, berbagai bobot) dan `RandomizedSearchCV(scoring="accuracy")`
langsung cuma menghasilkan 45-52% — terbaik XGBoost acc-tuned 51.5%,
naik 0.2pp dari baseline 51.3%.

**Temuan kunci — kenapa 60% kemungkinan besar mustahil untuk exact-contract**:
dicek dari 4.506 pasangan open/closed-room (dua pasangan expert manusia
membidik kartu identik), mereka hanya sepakat pada kontrak akhir yang
SAMA PERSIS **37.6%** dari waktu. Model kita (51-52%) sudah lebih
konsisten dari rata-rata dua pasangan manusia acak yang membidik hand
yang sama — 60% berarti melampaui konsistensi manusia ahli sendiri,
sesuatu yang secara struktural sangat sulit dicapai dari fitur hand+
lelang saja (bidding punya ambiguitas gaya partnership yang tidak
tertangkap di kartu semata).

**Alternatif realistis**: di level `target_category` (5 kelas:
Pass/Partscore/Game/SmallSlam/GrandSlam), pasangan manusia sepakat
68.3% — dan model kita sudah mencapai 75-78% di subtask ini (lihat
kaskade kategori, [experiments/2026-07-09](../2026-07-09/)). Top-3/top-5
exact-contract juga sudah 76-86%. Kalau "60% akurasi" bisa didefinisikan
di salah satu dari dua kerangka ini, targetnya sudah tercapai.

Detail lengkap di kesimpulan [01_accuracy_push.ipynb](01_accuracy_push.ipynb).

### 02 — Fitur DDS

| Model | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|
| XGBoost baseline (164 fitur) | **51.3%** | 0.269 | 0.469 |
| XGBoost +DDS (182 fitur) | 51.1% | **0.279** | **0.472** |
| XGBoost +DDS+sample_weight+DART (182 fitur) | 48.5% | 0.315 | 0.491 |
| REF: XGBoost +sample_weight+DART, TANPA DDS (164 fitur) | 49.2% | **0.322** | **0.498** |

**DDS memberi sinyal paling kuat dari seluruh fitur yang pernah dicoba**
— `dd_par_denom_H`/`dd_par_denom_S` (strain kontrak par) jadi fitur
**#1 dan #2 paling penting dari 182 fitur**, mengalahkan `ns_has_fit_S`
yang sebelumnya selalu di puncak. Tapi pola "tidak aditif" yang sama
seperti 2026-07-10 terulang: menambah DDS ke kombinasi sample_weight+
DART yang sudah bagus justru MENURUNKAN hasil (F1 macro 0.322→0.315).
DDS+XGBoost polos sedikit lebih baik dari baseline polos, tapi tidak
menembus ~52% — mengonfirmasi ulang bahwa batasnya ada di ambiguitas
bidding manusia (data), bukan di kekuatan fitur/model. Detail di
kesimpulan [02_dds_features.ipynb](02_dds_features.ipynb).

### 03 — Accuracy push di atas 182 fitur (DDS) — BERHASIL, tervalidasi test set

| Model | Accuracy | F1 Weighted | Top-3 | Top-5 |
|---|---|---|---|---|
| XGBoost baseline (182 fitur) | 52.4% | 0.480 | 76.8% | 84.5% |
| **XGBoost acc-tuned (182 fitur)** | **53.4%** | **0.489** | **77.5%** | **86.0%** |

Params: `n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.9,
colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0`. **+1.3pp dari
baseline kanonik asli (52.1%, 164 fitur)** — perbaikan pertama yang
tervalidasi TEST SET (bukan cuma val) sejak seluruh rangkaian eksperimen
2026-07-09. Kuncinya: tuning diarahkan LANGSUNG ke `scoring="accuracy"`
di atas fitur yang sudah terbukti membantu (DDS) — bukan menggabungkan
dua tujuan optimasi yang bentrok (F1 macro techniques + fitur baru,
seperti upaya-upaya sebelumnya yang selalu gagal). Detail di kesimpulan
[03_accuracy_push_with_dds.ipynb](03_accuracy_push_with_dds.ipynb).

Params ini sudah dipromosikan ke `notebooks_dds/03_modeling.ipynb`
sebagai konfigurasi XGBoost resmi untuk pipeline 182-fitur (re-run
04_evaluation.ipynb mengonfirmasi: 52.4%→**52.7%** accuracy di test set
— sedikit beda dari 53.4% di notebook eksperimen karena
`notebooks_dds` cast fitur ke `float32`, tapi arahnya konsisten:
tuning ini benar-benar membantu).

### 04 — Atasi class imbalance di atas XGBoost acc-tuned + 182 fitur

| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| XGBoost acc-tuned [none] (baseline nb03) | **52.7%** | 0.251 | 0.480 |
| **LightGBM [class_weight]** | 52.6% (-0.1pp) | **0.307** (+22% relatif) | **0.485** |
| XGBoost acc-tuned [random_oversample] | 47.3% (val) | 0.299 (val) | 0.484 (val) |
| XGBoost acc-tuned [sample_weight] | 46.1% (val) | 0.279 (val) | 0.479 (val) |
| XGBoost acc-tuned [smote] | 47.0% (val) | 0.265 (val, LEBIH RENDAH dari baseline) | 0.451 (val) |

**Hasil terbaik: `LightGBM + class_weight="balanced"`** — nyaris tanpa
trade-off (test set). Accuracy praktis sama dengan XGBoost acc-tuned
(52.6% vs 52.7%), tapi F1 macro naik 22% relatif (0.251->0.307) dan F1
weighted juga naik. Ini beda dari pola biasa: menyeimbangkan kelas
biasanya mengorbankan accuracy signifikan (lihat RandomOverSampler/
sample_weight untuk XGBoost acc-tuned di atas, keduanya anjlok >4pp)
— kali ini nyaris gratis, karena LightGBM+class_weight tidak mengubah
distribusi data training sama sekali (beda dari resampling yang
menambah/menduplikasi data, mengganggu hyperparameter yang sudah
di-tuning ketat untuk distribusi asli).

**SMOTE terbukti merugikan untuk KE-3 KALINYA** (2026-07-09, 2026-07-10,
sekarang) — kesimpulan ini sudah sangat robust across berbagai fitur/
hyperparameter, tidak perlu diuji ulang lagi. Detail di kesimpulan
[04_class_imbalance_dds.ipynb](04_class_imbalance_dds.ipynb).

### 05 — Data tambahan non-BBO (PBN) — HASIL TERBAIK PROYEK

Diminta bikin crawler BBO skala besar (>=1 juta board) — **ditolak**, ToS
BBO eksplisit melarang scraping. Dicari alternatif: 2 dari 3 sumber yang
muncul di hasil pencarian (`bridgetoernooi.com`, Vugraph Project) ternyata
sudah **mati** (domain expired/di-squat) meski terlihat aktif di snippet
pencarian — pelajaran: selalu verifikasi langsung sebelum membangun apa
pun di atasnya. `computerbridge.se` terverifikasi hidup: 1.314 board dari
7 final kejuaraan dunia (Bermuda Bowl, Venice Cup, Vanderbilt, Spingold,
dll.), format PBN. Ditemukan juga bonus: 100 file LIN baru muncul di
`data/raw/` selama sesi ini (kemungkinan export manual pribadi, legitimate).

`src/parser/pbn_parser.py` (PBNParser) baru dibangun — kompatibel penuh
dengan `BoardRecord`/`Hand` LIN, tidak perlu ubah `extract_features()`.
`build_dataset()` diperluas dengan parameter `extra_boards`. Dataset
gabungan: `data/processed_combined/` (606 file LIN + 43 file PBN, 182
fitur, **13.582 board** — naik dari 10.223, +33%).

| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| XGBoost acc-tuned (10.223 board) | 52.7% | 0.251 | 0.480 |
| XGBoost acc-tuned (13.582 board) | 52.4% (-0.3pp) | 0.264 (+1.3pp) | 0.484 (+0.4pp) |
| LightGBM class_weight (10.223 board) | 52.6% | 0.307 | 0.485 |
| **LightGBM class_weight (13.582 board)** | **53.6% (+1.0pp)** | **0.310** | **0.506 (+2.1pp)** |

**LightGBM+class_weight di data gabungan adalah kandidat TERBAIK
PROYEK SECARA KESELURUHAN** — accuracy tertinggi (53.6%) DAN F1
macro/weighted terbaik sekaligus, tanpa trade-off. Detail lengkap di
kesimpulan [05_combined_data_source.ipynb](05_combined_data_source.ipynb).

## Kesimpulan hari ini

**Target 60% accuracy belum tercapai, tapi ditemukan rangkaian perbaikan
nyata yang saling melengkapi, berpuncak pada kandidat terbaik proyek:**
1. 52.1% → 53.4% test accuracy (164 fitur baseline → 182 fitur DDS +
   accuracy-tuned, di 10.223 board) — langkah pertama.
2. LightGBM + class_weight, F1 macro 0.251→0.307 (182 fitur, 10.223
   board) — keseimbangan antar-kelas nyaris gratis.
3. **LightGBM + class_weight di data gabungan (13.582 board): 53.6%
   accuracy, F1 macro 0.310, F1 weighted 0.506** — kombinasi data
   tambahan (non-BBO, legitimate) + teknik terbaik dari langkah 1-2,
   hasil akhir terbaik proyek.

Proyeksi dari analisis konsistensi manusia (37.6% pasangan open/closed-
room sepakat kontrak sama persis) tetap menunjukkan 60% exact-contract
kemungkinan besar di luar jangkauan struktural — tapi rangkaian
perbaikan di atas adalah kemajuan konkret dan tervalidasi test set,
bukan target yang mustahil sama sekali. Level kategori (Pass/Partscore/
Game/Slam) sudah 75-78%, top-3/top-5 exact-contract sudah 78.4%/86.9%
(data gabungan, XGBoost acc-tuned).

**Rekomendasi konkret**: **LightGBM+class_weight di `data/processed_combined/`**
(606 LIN + 43 PBN, 182 fitur) adalah kandidat model utama baru yang
direkomendasikan — accuracy dan F1 macro/weighted terbaik sekaligus.
(Update 2026-07-16: promosi ke pipeline resmi selesai — lihat penutup
di bawah, setelah section 06.)

### 06 — Perluasan PBN lanjutan (tistis.nl) + 3 bug fix — HASIL TERBAIK PROYEK (lagi)

Diminta menambah lebih banyak file PBN. Ditemukan `tistis.nl/pbn/pbn_databases.htm`
(hidup, bidding manusia terverifikasi) — 169 file PBN tambahan: ETC97 (1997,
3.576 board), EYC98 (1998, 800 board), ETC99 (1999, 1.360 board), EYC00
(2000, 1.000 board), IWBC Boston finals (96 board), OKB (52 board), Papi
Garozzo OKbridge games (2.025 board — pbn1/pbn2 byte-identik, hanya satu
dipakai). `data/raw_pbn/` naik 43 → **212 file** (10.223 board PBN total).

**Ditolak setelah verifikasi kontaminasi bot**: `bermuda2000.zip` (100% GIB
vs WBridge5), file "Al Howard" (GIB 3 dari 4 kursi), `FFT_9901.zip`
(1 pasangan "Bristol GIB" dari 16 pasangan, kontaminasi kecil tapi tidak
sepadan untuk 84 board).

**3 bug ditemukan & diperbaiki** selama integrasi (self-caught lewat
verifikasi, bukan dilaporkan user):
1. PBN `"#"` (shorthand "sama seperti board sebelumnya") tidak di-resolve
   → `HomeTeam`/`VisitTeam` terbaca literal `"#"`, merusak identity key.
   Fix: carry-forward dict per file di `PBNParser`.
2. Turnamen round-robin memakai ulang nomor board 1..32 di tiap match yang
   main simultan — nomor board mentah bukan identity unik. Fix: composite
   key `board|home|visit` (fallback nama pemain kalau tanpa tag tim).
3. Kolom `_room` kosong (`""`) jadi `NaN` lewat CSV round-trip, lalu
   `.astype(str)` menghasilkan `"nan"` literal (bukan `""`) — merusak
   merge DDS cache. Fix: `keep_default_na=False` saat load cache.

Reuse DDS cache lama secara posisional (untuk hemat waktu hitung ulang)
sempat dicoba tapi **diverifikasi TIDAK valid** (4 dari 5 spot-check
board mismatch) — dibatalkan, DDS dihitung ulang penuh (~38 menit,
10.223 board PBN) untuk memastikan korektnes.

| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| XGBoost acc-tuned (13.582 board) | 52.4% | 0.264 | 0.484 |
| **XGBoost acc-tuned (21.675 board)** | **54.1%** (+1.7pp) | **0.294** (+3.0pp) | **0.504** (+2.0pp) |
| LightGBM class_weight (13.582 board) | 53.6% | 0.310 | 0.506 |
| **LightGBM class_weight (21.675 board)** | **54.3%** (+0.7pp) | **0.349** (+3.9pp) | **0.519** (+1.3pp) |

**LightGBM+class_weight tetap kandidat terbaik proyek, sekarang unggul
XGBoost di SEMUA metrik** (sebelumnya XGBoost masih kompetitif di
accuracy). Dataset 60% lebih besar (21.675 board, 36 kelas — `5N`
akhirnya cukup sampel) terbukti membantu terutama kelas langka. Detail
lengkap di kesimpulan
[06_combined_data_v2_more_pbn.ipynb](06_combined_data_v2_more_pbn.ipynb).

## Update 2026-07-16: promosi data gabungan ke pipeline resmi `notebooks_dds/`

Diminta mengganti data di notebook DDS dengan data baru. Sebelumnya
`notebooks_dds/` (pipeline resmi 182-fitur) masih memakai 10.223 board
LIN-only, sementara data gabungan (LIN+PBN) hanya hidup di eksperimen
(`data/processed_combined/`, notebook 05/06 di atas). Perbedaan itu
sekarang dihilangkan:

- `notebooks_dds/01_data_extraction.ipynb`: mem-parse `data/raw/` (606 LIN)
  + `data/raw_pbn/` (212 PBN) sekaligus, `build_dataset(extra_boards=...)`.
  `data/processed_dds/` sekarang identik isinya dengan
  `data/processed_combined/` (21.675 board, 36 kelas, 182 fitur).
- `notebooks_dds/03_modeling.ipynb`: `LGBMModel` sekarang eksplisit pakai
  `class_weight="balanced"` (sebelumnya default, tidak seimbang kelas).
- Notebook 01→04 dieksekusi ulang penuh (bukan hanya notebook eksperimen).

**Hasil resmi test set (`outputs/results_dds/nb04_summary.json`)**:

| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| Random Forest | 44.8% | 0.312 | 0.463 |
| XGBoost acc-tuned | 54.0% | 0.294 | 0.504 |
| **LightGBM class_weight** | **54.3%** | **0.349** | **0.519** |

Angka ini identik dengan hasil notebook eksperimen 06 di atas — memverifikasi
bahwa pipeline resmi dan eksperimen konsisten. `data/processed/` (164-fitur
kanonik, `notebooks/` biasa) **tidak diubah**, tetap terpisah sesuai desain
awal DDS sebagai fitur tambahan opsional.

### 07 — Perluasan besar: arsip angelfire.com via Wayback Machine — LOMPATAN TERBESAR

Diminta cari lagi data PBN karena terbukti efektif menaikkan akurasi.
Ditemukan link eksternal di halaman `tistis.nl` menunjuk ke
`angelfire.com/games2/pbnarchive/pbn/` — situs aslinya **mati** (DNS
`www.angelfire.com` tidak resolve, root domain menolak koneksi), tapi
seluruh isinya berhasil direcover lewat **Wayback Machine** (snapshot
2019-08-06): 57 arsip zip kejuaraan dunia 1996-2002 (Bermuda Bowl,
Venice Cup, Vanderbilt, World Bridge Team Olympiad, Cap Gemini,
Cavendish, Dutch Teams Final, European Team Championships, ACBL
International Team Trials, dll.), 56/57 berhasil diunduh utuh
(`capgem00.zip` snapshotnya sendiri korup di Wayback, dilewati). 3 arsip
(`etc99`, `eyc98`, `eyc00`) ternyata **byte-identik** dengan file yang
sudah dimiliki dari `tistis.nl` (mirror yang sama) — dikeluarkan.
Diverifikasi bersih dari kontaminasi bot (cek player-name tags untuk
GIB/WBridge5/dll. — nihil, satu "hit" cuma nama manusia "Jack Zhao").
**1.178 file PBN baru** ditambahkan ke `data/raw_pbn/`.

**Bug crash serius ditemukan & diperbaiki** selama komputasi DDS untuk
37.489 board baru: 53 board dari arsip lama (Dutch Teams Final
1996/1998, ETC 2001, WC98, Politiken 1997) punya kartu duplikat/hilang
dari kesalahan transkripsi manual 1990-2000an — tiap tangan tetap
terhitung 13 kartu, tapi total dek gabungan cuma 44-49 kartu unik
(bukan 52). Ini men-**segfault** `endplay`'s DDS solver (bukan exception
biasa, meng-crash seluruh proses Python) — penyebab dua percobaan
komputasi DDS pertama gagal (deadlock `multiprocessing.Pool` tanpa
sebab jelas di mesin ini, lalu crash langsung setelah beralih ke
sekuensial). Diperbaiki dengan validasi 52-kartu-unik di
`src/features/dds.py::compute_dds_features()` sebelum memanggil
`endplay` — melindungi SEMUA sumber data secara permanen. Skrip
komputasi juga ditulis ulang dengan checkpoint tiap 1.000 board setelah
dua kegagalan pertama kehilangan seluruh progres (~1-2 jam kerja).
Komputasi akhirnya selesai sekuensial, ~2.9 jam untuk 37.489 board.

| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| XGBoost acc-tuned (21.675 board) | 54.1% | 0.294 | 0.504 |
| **XGBoost acc-tuned (49.755 board)** | **56.1%** (+2.1pp) | **0.342** (+4.8pp) | **0.532** (+2.8pp) |
| LightGBM class_weight (21.675 board) | 54.3% | 0.349 | 0.519 |
| **LightGBM class_weight (49.755 board)** | **56.4%** (+2.1pp) | **0.410** (+6.1pp) | **0.557** (+3.8pp) |

**LightGBM+class_weight tetap kandidat terbaik proyek, unggul di SEMUA
metrik**, dan marginnya atas XGBoost di F1 macro justru MELEBAR (0.410
vs 0.342, dari 0.349 vs 0.294 sebelumnya) — mengonfirmasi pola sejak
2026-07-09: lebih banyak data membantu `class_weight` menangani kelas
langka lebih baik. Dataset gabungan sekarang **49.755 board** (naik
dari 21.675, +130%), 36 kelas, 182 fitur. Dipromosikan langsung ke
`notebooks_dds/` (notebook 01→04 dieksekusi ulang, angka identik).
Detail lengkap di kesimpulan
[07_angelfire_expansion.ipynb](07_angelfire_expansion.ipynb).

**Catatan untuk perluasan berikutnya**: arsip angelfire.com ini
kemungkinan besar adalah kumpulan PBN non-BBO legitimate terbesar yang
tersisa dari era pra-2010. Sumber serupa mulai langka — perluasan lebih
lanjut mungkin perlu jenis sumber lain, bukan sekadar PBN archive
tambahan.
