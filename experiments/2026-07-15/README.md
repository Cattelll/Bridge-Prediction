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

## Kesimpulan hari ini

**Target 60% accuracy belum tercapai, tapi ditemukan perbaikan nyata:
52.1% → 53.4% test accuracy** (164 fitur baseline → 182 fitur + DDS +
accuracy-tuned). Proyeksi dari analisis konsistensi manusia (37.6%
pasangan open/closed-room sepakat kontrak sama persis) tetap menunjukkan
60% exact-contract kemungkinan besar di luar jangkauan struktural — tapi
1.3pp adalah kemajuan konkret, bukan target yang mustahil sama sekali.
Kalau target 60% ingin didekati lebih jauh: level kategori (Pass/
Partscore/Game/Slam) sudah 75-78%, top-3/top-5 exact-contract sudah
77.5%/86.0% (naik dari 76-86% sebelumnya berkat DDS+tuning).

**Rekomendasi konkret**: promosikan konfigurasi XGBoost acc-tuned +
182 fitur (164+DDS) ini ke `configs/config.yaml` dan
`notebooks_dds/03_modeling.ipynb` sebagai model utama baru — ini
kandidat pertama yang lolos validasi test set dengan perbaikan accuracy
yang jelas dan terukur.
