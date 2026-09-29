# Resume Semua Ringkasan Proyek

*Dibuat 2026-09-24, angka disinkronkan 2026-09-27 (definisi label diperketat). Rangkuman dari seluruh dokumen ringkasan di repo ini:
apa isi masing-masing, mana yang masih berlaku, dan apa kesimpulan bersamanya.*

---

## 1. Peta dokumen ringkasan

| Dokumen | Isi | Status |
|---|---|---|
| [CLAUDE.md](../CLAUDE.md) | Riwayat lengkap, batas ruang lingkup, invariant | ✅ Terkini |
| [ringkasan.md](../ringkasan.md) | Ringkasan satu halaman + tabel riwayat | ✅ Terkini |
| [README.md](../README.md) | Halaman depan repo | ✅ Terkini (ditulis ulang 2026-09-24) |
| [docs/SUMMARY.md](SUMMARY.md) | Ringkasan untuk pembaca baru | ✅ Terkini (target biner) |
| [docs/EXPERIMENTS.md](EXPERIMENTS.md) | Alur ablasi fitur DD & Exp A/B/C/E/F | ✅ Terkini |
| [docs/ALGORITHMS.md](ALGORITHMS.md) | Teori RF vs XGBoost vs LightGBM + kaitan ke hasil | ✅ Terkini |
| [outputs/results/REPORT.md](../outputs/results/REPORT.md) | Laporan otomatis `scripts/report.py` | ✅ Terkini (metrik biner, 8 kandidat — 2026-09-27) |
| [docs/FEATURES.md](FEATURES.md) | Kamus 164 fitur kanonik | ⚠️ 10 fitur trik DD belum didokumentasikan |
| [docs/ARCHITECTURE.md](ARCHITECTURE.md) | Alur pipeline & modul | ⚠️ Masih menyebut metrik top-k |
| `experiments/2026-07-09`, `07-10`, `07-15`, `08-31/README.md` | Log eksperimen | 📦 Arsip era 36 kelas |
| `outputs/results/nb04/05/06_summary.json` | Angka mentah | ✅ Sumber angka resmi |

---

## 2. Inti proyek saat ini

- **Tujuan**: membandingkan Random Forest, XGBoost, dan LightGBM untuk
  memprediksi apakah kontrak yang dibid manusia cocok dengan salah satu
  **kontrak par** Double-Dummy Solver (level, strain, status double, dan
  sisi declarer — diperketat 2026-09-27).
- **Target**: `matches_par_contract` (biner) — 24,3% cocok / 75,7% tidak.
  Menggantikan target 36 kelas sejak 2026-09-18.
- **Data**: 606 file LIN (BBO) + 1.390 file PBN (computerbridge.se, tistis.nl,
  angelfire.com via Wayback Machine) → **49.755 board**, split 70/15/15
  group-aware (34.829 / 7.462 / 7.464).
- **Fitur**: **174** = 164 kanonik + 10 trik DD per strain. Ablasi: trik DD
  menaikkan F1 macro val +5,40pp.
- **Anti-leakage**: kolom par DDS (8 `dd_par_*` + `dd_par_contracts`) dikeluarkan dari fitur (bahan pembentuk
  label). Pasangan open/closed-room BBO selalu satu split.

---

## 3. Hasil test set

| Model | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Baseline RandomForest | 82,3% | 0,754 | 0,821 | 0,823 | 0,667 |
| Baseline XGBoost | 83,0% | 0,729 | 0,814 | 0,831 | 0,677 |
| **Baseline LightGBM** | 81,5% | 0,761 | 0,819 | 0,845 | 0,702 |
| Exp A — XGBoost retuned | **84,9%** | 0,769 | 0,839 | 0,848 | 0,709 |
| Exp B — XGBoost balanced penuh | 83,2% | 0,773 | 0,832 | 0,847 | 0,709 |
| Exp C — LightGBM retuned | 83,1% | 0,775 | 0,832 | **0,849** | **0,712** |
| Exp E — RandomForest retuned | 82,9% | 0,766 | 0,829 | 0,840 | 0,700 |
| **Exp F — XGBoost soft-balanced α=0,5** | 84,7% | **0,782** | **0,843** | **0,849** | 0,707 |

Baseline mayoritas = 75,7% accuracy; semua model mengunggulinya. ROC-AUC
~0,82–0,85 → tidak ada tanda leakage residual.

---

## 4. Kesimpulan bersama

1. **XGBoost Exp F = kandidat terbaik keseluruhan** — F1 macro tertinggi dan
   accuracy hanya 0,2pp di bawah Exp A. **Belum dipromosikan** ke baseline resmi.
2. **LightGBM = baseline resmi terbaik** (F1 macro & ROC-AUC tertinggi di
   antara baseline).
3. **Trik DD sangat membantu; kolom par DDS = leakage** (tidak boleh jadi fitur).
4. **Sirkularitas fitur lelang belum ditindaklanjuti** — fitur `auction_*`
   dihitung dari lelang yang sudah selesai dan menduduki 3 teratas SHAP
   (eksperimen 2026-08-31). Disarankan ablasi eksplisit.

---

## 5. Pelajaran dari arsip eksperimen (era 36 kelas)

| Tanggal | Pelajaran |
|---|---|
| 2026-07-09 | Split acak bocor (~60% baris val/test punya kembaran tangan identik di train) → split group-aware. SMOTE & GOSS merugikan. |
| 2026-07-10 | Teknik yang membantu secara terpisah **tidak otomatis aditif** saat digabung — kombinasi justru sedikit menurun. |
| 2026-07-15/16 | **Menambah data PBN = perbaikan terbesar** (accuracy 52%→56,4%, F1 macro 0,31→0,41). Crawler BBO ditolak (ToS). Bug segfault DDS pada dek ≠52 kartu diperbaiki. |
| 2026-08-31 | Seleksi fitur SHAP: top-40 fitur setara 182 fitur — gain dalam noise, tidak dipromosikan. |

---

## 6. Pekerjaan tersisa

- `scripts/run_pipeline.py` masih membangun pipeline lama — **jangan
  dijalankan** sampai diupdate mengikuti `notebooks/01`.
- `docs/ARCHITECTURE.md` (metrik top-k) dan `docs/FEATURES.md` (10 fitur trik
  DD) perlu disinkronkan.
- Keputusan promosi XGBoost Exp F ke baseline resmi belum diambil.
- Ablasi fitur lelang (`auction_*`) untuk menguji sirkularitas.
