# Percobaan 2026-07-10

Lanjutan dari [experiments/2026-07-09/](../2026-07-09/) setelah perbaikan
kebocoran split (group-aware, lihat catatan di README eksperimen kemarin).
Fokus: **menggabungkan** hanya teknik yang terbukti membantu, bukan
menumpuk kelima eksperimen kemarin mentah-mentah.

## Daftar percobaan

| # | Notebook | Tujuan | Status |
|---|----------|--------|--------|
| 1 | [01_combined_best_techniques.ipynb](01_combined_best_techniques.ipynb) | Gabungkan 23 fitur tambahan (nb04) + sample_weight/RandomOverSampler + DART (nb05), plus dicoba dengan params tuning XGBoost (nb01) | done |

## Aturan main

Sama seperti [experiments/2026-07-09/README.md](../2026-07-09/README.md):
split `data/processed/` dipakai apa adanya (group-aware, seed 42,
70/15/15), val set untuk membandingkan varian, test set disentuh sekali
di akhir untuk kandidat final.

## Kenapa tidak menggabungkan semua 5 teknik kemarin

| Teknik | Diikutkan? | Alasan |
|---|---|---|
| Fitur tambahan (nb04) | Ya | Perbaikan kecil tapi konsisten setelah split diperbaiki |
| sample_weight / RandomOverSampler (nb02, nb05) | Ya (dua-duanya dicoba) | Dua-duanya solid; belum jelas mana yang lebih baik dikombinasikan dengan DART |
| DART (nb05) | Ya | Komponen kandidat terbaik & paling robust kemarin |
| Hyperparameter tuning (nb01) | Dicoba dengan catatan | Params ditemukan di ruang pencarian `gbtree`/164 fitur — dipakai ulang sebagai pendekatan, bukan re-tuning baru |
| SMOTE (nb02) | Tidak | Konsisten merugikan di kedua split |
| GOSS (nb05) | Tidak | Konsisten merugikan LightGBM |
| Two-stage cascade (nb03) | Tidak | Keuntungan menyusut banyak setelah split diperbaiki; sulit digabung bersih dengan fitur tambahan |

## Ringkasan hasil

**Jawaban singkat untuk "apakah perlu digabungkan semua metode": tidak
— bahkan menggabungkan yang sudah terbukti membantu secara terpisah
JUSTRU SEDIKIT MENURUNKAN hasil di sini.**

| Model | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|
| **164 fitur + sample_weight + DART** (anchor, dari 2026-07-09) | 49.2% | **0.322** | **0.498** |
| 187 fitur + RandomOverSampler + DART | 49.2% | 0.307 | 0.494 |
| 187 fitur + sample_weight + DART | 48.7% | 0.315 | 0.493 |
| 187 fitur + sample_weight + DART + tuned params (nb01) | 46.9% | 0.313 | 0.484 |
| XGBoost baseline (164 fitur, tanpa teknik apa pun) | 51.3% | 0.269 | 0.469 |

Tidak satu pun kombinasi mengalahkan kandidat tunggal terbaik kemarin
(164 fitur + sample_weight + DART). Menambahkan 23 fitur tambahan ke
kombinasi yang sudah bagus ini justru MENURUNKAN F1 macro — kebalikan
dari efeknya saat ditambahkan ke XGBoost polos (nb04, 2026-07-09).
Pelajaran utama: **perbaikan yang valid secara terpisah tidak otomatis
aditif ketika digabung** — harus selalu diuji ulang sebagai kombinasi.
Detail & analisis lengkap di kesimpulan
[01_combined_best_techniques.ipynb](01_combined_best_techniques.ipynb).

**Rekomendasi**: hentikan kombinasi lebih lanjut untuk arah ini;
validasikan "164 fitur + sample_weight + DART" (murni, tanpa tambahan
apa pun) di **test set** sebagai kandidat model utama berikutnya.
