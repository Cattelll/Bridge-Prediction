# Ringkasan Proyek

Satu halaman untuk siapa pun yang baru masuk ke repo ini. Untuk detail lebih
dalam lihat [ARCHITECTURE.md](ARCHITECTURE.md) (alur pipeline & modul) dan
[FEATURES.md](FEATURES.md) (data dictionary 164 fitur).

## Tujuan

Penelitian skripsi yang membandingkan tiga algoritma ensemble berbasis pohon
— Random Forest, XGBoost, LightGBM — untuk memprediksi **kontrak optimal**
dalam Contract Bridge (level + strain, mis. `"4S"`, `"3N"`, atau `"PASS"`)
dari data hasil lelang (bidding) dan kartu, direkam dalam format BBO LIN.
Ini adalah masalah klasifikasi multikelas dengan 35 kelas kontrak.

## Dataset

| Item | Nilai |
|---|---|
| Sumber | 506 file `.lin` dari Bridge Base Online |
| Papan setelah dedup & pembersihan | 10.223 |
| Split (train / val / test, **group-aware**) | 7.157 / 1.533 / 1.533 (70/15/15) |
| Fitur | 164 (numerik/biner — lihat [FEATURES.md](FEATURES.md)) |
| Kelas target (`target_base`) | 35 kontrak + implisit PASS |
| Random seed | 42 (tetap di semua tahap acak) |

> **Split group-aware (2026-07-09)**: BBO vugraph mencatat tiap papan dua
> kali (open room/closed room — dua pasangan membidik kartu yang **sama
> persis**). Split acak biasa membiarkan ~46% pasangan itu terpecah ke
> partisi berbeda, jadi ~60% baris val/test punya "kembaran" tangan
> identik di train — kebocoran yang bikin sebagian akurasi berasal dari
> hafalan pasangan, bukan generalisasi murni (diverifikasi: akurasi
> 92-95% saat kembarannya di train bid kontrak SAMA, vs 23-28% saat
> BEDA). Sekarang dikelompokkan per papan fisik (`StratifiedGroupKFold`
> pada `_source_file`+`_board_number`) di `build_dataset()`. Detail &
> angka before/after: [experiments/2026-07-09/README.md](../experiments/2026-07-09/README.md).

## Metode

```
LIN files → LINParser → BoardRecord
          → extract_features() → 164 fitur + 3 kolom target
          → build_dataset() → dedup, encode label, split 70/15/15
          → RFModel / XGBModel / LGBMModel .fit()
          → evaluate() → accuracy, F1 macro/weighted, top-k, confusion matrix
```

Detail lengkap di [ARCHITECTURE.md](ARCHITECTURE.md).

## Hasil (test set)

*Retrain terakhir: 2026-07-09 (sore, setelah perbaikan split group-aware),
di atas dataset 506 file / 10.223 papan, `n_estimators=300` di ketiga model
— lihat [outputs/results/test_comparison.csv](../outputs/results/test_comparison.csv)*

| Model | Accuracy | Top-3 | Top-5 | F1 Macro | F1 Weighted |
|---|---|---|---|---|---|
| RandomForest | 44.9% | 71.8% | 82.0% | 0.278 | 0.465 |
| **XGBoost** | **52.1%** | **76.3%** | **85.1%** | **0.290** | **0.481** |
| LightGBM | 51.7% | 74.6% | 84.0% | 0.279 | 0.467 |

> Dengan split lama (bocor lewat pasangan open/closed-room), angkanya sedikit
> berbeda (46.3/52.9/51.7% accuracy, LightGBM sempat unggul F1 Macro). Efek
> kebocorannya sebagian saling meniadakan di level agregat, jadi perubahan
> angka relatif kecil (±1-2pp) — tapi metodologinya sekarang jauh lebih
> defensible. Detail investigasi di [experiments/2026-07-09/README.md](../experiments/2026-07-09/README.md).

## Temuan Utama

- **XGBoost unggul di SEMUA metrik** dengan split yang diperbaiki (accuracy,
  top-k, F1 macro, F1 weighted) — pilihan terbaik keseluruhan.
- **LightGBM** sedikit di bawah XGBoost di semua metrik kali ini (F1 Macro
  sebelumnya sempat unggul dengan split lama yang bocor — temuan itu tidak
  bertahan setelah split diperbaiki).
- Fitur paling berpengaruh berbeda antar model: RandomForest & LightGBM masih
  didominasi `auction_len`/`auction_ns_bids`/`auction_ew_bids` (fitur ringkasan
  lelang yang baru diketahui setelah lelang SELESAI — perlu dibaca hati-hati,
  lihat catatan sirkularitas di bawah), sedangkan **XGBoost** didominasi fitur
  kekuatan tangan yang lebih genuine: `ns_has_fit_S/H`, `ew_has_fit_S/H`,
  `ns_hcp`/`ew_hcp` — sejalan dengan teori bidding bridge bahwa fit &
  kekuatan partnership adalah prediktor kontrak paling kuat.
- F1 Macro jauh lebih rendah dari F1 Weighted/accuracy di ketiga model —
  konsekuensi class imbalance yang berat (kontrak slam sangat jarang
  dibanding partscore).

### Catatan sirkularitas fitur lelang (temuan sekunder, belum ditindaklanjuti)

Fitur `auction_len`, `auction_ns_bids`, `auction_ew_bids`, `auction_doubled`,
dkk. dihitung dari transkrip lelang yang **sudah selesai** — pada saat itu
kontrak final sudah ditentukan, jadi ada sirkularitas konseptual (memakai
ringkasan hasil lelang untuk memprediksi hasil lelang itu sendiri). Feature
importance menunjukkan dampaknya sedang: dominan di RF/LightGBM, tapi bukan
fitur teratas di XGBoost (fitur kekuatan tangan lebih dominan di sana). Belum
diperbaiki/didiskusikan lebih lanjut — dicatat di sini supaya tidak terlewat.

## Batasan & Status Saat Ini

- **Notebook 01–04 belum dieksekusi ulang** di dataset 506 file yang
  diperbesar ini. Tabel di atas berasal dari `scripts/run_pipeline.py`
  (dijalankan langsung, bukan lewat notebook), sehingga visualisasi PNG di
  `outputs/results/` dan `outputs/results/REPORT.md` (yang dibangkitkan dari
  `nb04_summary.json`) masih mencerminkan run lama (6.963 papan/156 fitur).
  Jalankan ulang notebook 01→04 secara berurutan untuk menyamakan semuanya.
- `pyproject.toml` saat ini terhapus di working tree lokal — instalasi
  sementara pakai `requirements.txt` (lihat README).
- Lihat [CLAUDE.md](../CLAUDE.md) untuk batas ruang lingkup penelitian
  (algoritma/teknik apa yang sengaja tidak dipakai) dan invariant yang tidak
  boleh diubah tanpa alasan kuat (seed, urutan fitur, target utama, rasio
  split).
