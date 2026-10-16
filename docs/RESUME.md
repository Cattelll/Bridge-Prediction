# Resume Dokumentasi Proyek

Status sinkronisasi: 2026-10-16. Angka terbaru bersumber dari notebook 01?06 yang
dijalankan ulang setelah split global berdasarkan deal dan deduplikasi
fitur-label diperbaiki.

| Dokumen | Cakupan | Status |
|---|---|---|
| [README](../README.md) | Ringkasan repo dan hasil run terbaru | Terkini |
| [SUMMARY](SUMMARY.md) | Ringkasan penelitian | Terkini |
| [ARCHITECTURE](ARCHITECTURE.md) | Pipeline, label, split, evaluasi | Terkini |
| [FEATURES](FEATURES.md) | 164 fitur kanonik + 10 fitur DDS | Terkini |
| [EXPERIMENTS](EXPERIMENTS.md) | Baseline, seleksi, ablasi dan batasan | Terkini |
| [ALGORITHMS](ALGORITHMS.md) | Teori algoritma | Referensi konsep |
| [REPORT](../outputs/results/REPORT.md) | Metrik run terbaru | Terkini |
| `experiments/2026-07-*`, `2026-08-*` | Riwayat target lama/split sebelumnya | Arsip historis |

Target adalah `matches_par_contract`: kecocokan kontrak aktual dengan salah satu
kontrak par DDS v4 berdasarkan level, strain, double/redouble, dan sisi declarer.
Dataset berisi 50.227 baris, 174 fitur, serta prevalensi positif 24,56%.

Split memakai tanda tangan deal 52 kartu lintas semua sumber. Validation memilih
model/threshold; test hanya mengestimasi pilihan yang dikunci. Metrik AP/F1/ROC-AUC
dan precision/recall dilaporkan; AP berarti Average Precision dari scikit-learn.
