# Percobaan 2026-07-15

Target: naikkan **accuracy** (bukan F1 macro/weighted) mendekati 60%.
Lanjutan dari [experiments/2026-07-09/](../2026-07-09/) dan
[experiments/2026-07-10/](../2026-07-10/) — kedua sesi itu mentok di
~52% accuracy val set, dan sebagian teknik yang dicoba (sample_weight,
DART, resampling) justru menukar accuracy demi F1 macro.

## Daftar percobaan

| # | Notebook | Tujuan | Status |
|---|----------|--------|--------|
| 1 | [01_accuracy_push.ipynb](01_accuracy_push.ipynb) | Soft-voting ensemble RF+XGB+LGBM, dan `RandomizedSearchCV(scoring="accuracy")` (bukan f1_macro seperti nb01 2026-07-09) | done |
| 2 | [02_dds_features.ipynb](02_dds_features.ipynb) | Fitur Double-Dummy Solver (`endplay`) — DD tricks per strain + kontrak par. **Scope disetujui 2026-07-15**, lihat `CLAUDE.md` | done |

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

## Kesimpulan hari ini

**Target 60% accuracy tidak tercapai, dan kemungkinan besar secara
struktural tidak bisa dicapai** untuk exact-contract 35-kelas — dibatasi
oleh konsistensi bidding manusia sendiri (37.6%), bukan oleh kekuatan
model/fitur. DDS (baru disetujui masuk scope) memberi fitur paling
informatif yang pernah ditemukan (par contract strain, rank #1-#2),
tapi bahkan itu tidak cukup menembus ~52%. Kalau target realistis
diinginkan: level kategori (Pass/Partscore/Game/Slam) sudah 75-78%,
top-3/top-5 exact-contract sudah 76-86%.
