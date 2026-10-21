# Resume Dokumentasi Proyek

Status sinkronisasi: 2026-10-21. Pipeline membandingkan dataset DDS 174 fitur dan noDDS 164 fitur yang dibangun dari board, label, deduplikasi, dan split deal-aware yang sama. Notebook 04 melatih baseline, Notebook 05 mengevaluasi baseline, Notebook 06 menguji kandidat noDDS, Notebook 07 membandingkan improvement berpasangan, dan Notebook 08 mengevaluasi hasil final.

| Dokumen | Cakupan | Status |
|---|---|---|
| README | Ringkasan repo dan hasil tersimpan | Skor lama ditandai historis |
| SUMMARY | Ringkasan penelitian | Diselaraskan ke alur dataset berpasangan |
| ARCHITECTURE | Pipeline, label, split, evaluasi | Diselaraskan |
| FEATURES | 164 fitur kanonik + fitur DDS sumber | Data dictionary |
| EXPERIMENTS | Hasil run dan eksperimen terdahulu | Skor run 174 fitur bersifat historis |
| ALGORITHMS | Teori algoritma | Referensi konsep |
| REPORT | Metrik run tersimpan | Perlu diperbarui dari pipeline 164 fitur |

Label `matches_par_contract` menandai kecocokan kontrak aktual dengan salah satu kontrak par DDS v4 berdasarkan level, strain, status double/redouble, dan sisi declarer. DDS tetap diperlukan untuk membentuk label meski fitur jumlah trik DDS tidak digunakan dalam model utama.

Dataset berisi 50.227 baris; split deal-aware memiliki 35.161 train, 7.534 validation, dan 7.532 test. Notebook 07 menguji kandidat pada test setelah pemilihan berdasarkan validation; Notebook 08 merangkum evaluasi final.
