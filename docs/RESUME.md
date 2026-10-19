# Resume Dokumentasi Proyek

Status sinkronisasi: 2026-10-19. Pipeline utama memakai dataset 164 fitur kanonik tanpa fitur jumlah trik DDS. Notebook 03 membangun dataset 174 fitur dengan DDS dan dataset noDDS secara berpasangan dari board, label, deduplikasi, dan split yang sama. Modeling berjalan di Notebook 04, diikuti evaluasi baseline, eksperimen kandidat, dan evaluasi final pada Notebook 05–07.

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

Dataset berisi 50.227 baris; split deal-aware memiliki 35.161 train, 7.534 validation, dan 7.532 test. Skor final untuk pipeline 164 fitur belum tersedia sampai Notebook 04–07 dijalankan ulang berurutan.
