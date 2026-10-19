# Resume Dokumentasi Proyek

Status sinkronisasi: 2026-10-19. Notebook 04 menempatkan ablation fitur DDS setelah pemeriksaan data
ke rangkaian notebook penelitian stabil. Ia membandingkan 174 vs 164 fitur
pada label dan split yang sama.

| Dokumen | Cakupan | Status |
|---|---|---|
| [README](../README.md) | Ringkasan repo dan hasil run terbaru | Terkini |
| [SUMMARY](SUMMARY.md) | Ringkasan penelitian | Terkini |
| [ARCHITECTURE](ARCHITECTURE.md) | Pipeline, label, split, evaluasi | Terkini |
| [FEATURES](FEATURES.md) | 164 fitur kanonik + 10 fitur DDS | Terkini |
| [EXPERIMENTS](EXPERIMENTS.md) | Baseline, seleksi, ablasi dan batasan | Terkini |
| [ALGORITHMS](ALGORITHMS.md) | Teori algoritma | Referensi konsep |
| [REPORT](../outputs/results/REPORT.md) | Metrik run terbaru | Terkini |
| `experiments/2026-07-*`, `2026-08-*` | Riwayat eksperimen dan split terdahulu | Arsip historis |

Target adalah `matches_par_contract`: kecocokan kontrak aktual dengan salah satu
kontrak par DDS v4 berdasarkan level, strain, double/redouble, dan sisi declarer.
Dataset berisi 50.227 baris, 174 fitur, serta prevalensi positif 24,56%.

Split memakai tanda tangan deal 52 kartu lintas semua sumber. Validation memilih
model/threshold; test hanya mengestimasi pilihan yang dikunci. Metrik AP/F1/ROC-AUC
dan precision/recall dilaporkan; AP berarti Average Precision dari scikit-learn.
