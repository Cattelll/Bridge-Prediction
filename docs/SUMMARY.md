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
| Split (train / val / test, stratified) | 7.156 / 1.533 / 1.534 (70/15/15) |
| Fitur | 164 (numerik/biner — lihat [FEATURES.md](FEATURES.md)) |
| Kelas target (`target_base`) | 35 kontrak + implisit PASS |
| Random seed | 42 (tetap di semua tahap acak) |

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

*Retrain terakhir: 2026-07-09, di atas dataset 506 file / 10.223 papan,
`n_estimators=300` disamakan di ketiga model
— lihat [outputs/results/test_comparison.csv](../outputs/results/test_comparison.csv)*

| Model | Accuracy | Top-3 | Top-5 | F1 Macro | F1 Weighted |
|---|---|---|---|---|---|
| RandomForest | 46.3% | 75.7% | 85.8% | 0.244 | 0.448 |
| **XGBoost** | **52.9%** | **78.4%** | **86.6%** | 0.275 | **0.500** |
| LightGBM | 51.7% | 76.9% | 85.3% | **0.280** | 0.486 |

> Menaikkan RandomForest dari 200→300 trees (percobaan menyamakan hyperparameter
> antar model) tidak memberi perbaikan berarti — accuracy/F1 turun ~0.1pp,
> top-k naik ~0.5-1pp, level noise. RF tampaknya sudah konvergen di ~200 trees
> untuk dataset ini.

## Temuan Utama

- **XGBoost** unggul di accuracy, top-k, dan F1 weighted — pilihan terbaik
  jika prioritasnya prediksi tepat pada kelas-kelas umum (kontrak sering
  muncul seperti partscore/game).
- **LightGBM** unggul tipis di F1 Macro — sedikit lebih baik menangani kelas
  langka (small/grand slam) meski akurasi keseluruhan lebih rendah dari
  XGBoost.
- Fitur paling berpengaruh (feature importance, konsisten di ketiga model):
  `auction_len`, kombinasi HCP partnership (`ns_hcp`, `ew_hcp`,
  `hcp_ns_advantage`), dan indikator fit suit (`ns_has_fit_S/H`,
  `ew_has_fit_S/H`) — sejalan dengan teori bidding bridge bahwa panjang
  lelang dan kekuatan/fit partnership adalah prediktor kontrak paling kuat.
- F1 Macro jauh lebih rendah dari F1 Weighted/accuracy di ketiga model —
  konsekuensi class imbalance yang berat (kontrak slam sangat jarang
  dibanding partscore).

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
