# Arsitektur dan Alur Pipeline

Pipeline utama mengklasifikasikan kecocokan kontrak aktual dengan kontrak par
dari rekaman LIN dan PBN. DDS v4 membentuk label; fitur jumlah trik DDS tidak
dipakai dalam model utama.

## Alur penelitian aktif

```mermaid
flowchart TD
  A([Mulai]) --> B1[(Sumber LIN)]
  A --> B2[(Sumber PBN)]
  B1 --> C[Preprocessing dan normalisasi]
  B2 --> C
  C --> D[Ekstraksi 164 fitur kanonik]
  D --> E[(DDS v4 untuk label)]
  E --> F[Split data deal-aware<br/>satu assignment untuk kedua varian]
  F --> F1[(Dataset DDS: 174 fitur)]
  F --> F2[(Dataset noDDS: 164 fitur)]
  F2 --> H1[Random Forest]
  F2 --> H2[XGBoost]
  F2 --> H3[LightGBM]
  H1 --> I[Analisis validation dan interpretasi]
  H2 --> I
  H3 --> I
  I --> J[Evaluasi final pada test]
  J --> K([Selesai])
```

Notebook 03 menghitung label DDS, melakukan deduplikasi dan split deal-aware sekali, lalu menyimpan dua proyeksi fitur dengan assignment split yang sama. Dataset `data/processed/` menyimpan 174 fitur (164 kanonik + 10 trik DDS); dataset `data/processed_no_dds/` menyimpan 164 fitur kanonik tanpa kolom trik maupun rincian par DDS. Notebook 04 melatih baseline pada noDDS; Notebook 05 menganalisis hasil baseline, Notebook 06 menjalankan eksperimen, dan Notebook 07 melakukan evaluasi final.

## Label, fitur, dan split

Label matches_par_contract bernilai 1 bila kontrak aktual cocok dengan salah
satu kontrak par berdasarkan level, strain, status double/redouble, dan sisi
declarer. Kolom par yang langsung membentuk label dikeluarkan dari fitur.
Dataset tanpa fitur trik DDS tetap memakai label yang dihitung DDS.

Split dikelompokkan menurut identitas kanonik deal 52 kartu. Deal yang sama
tidak melintasi train, validation, dan test. Dataset aktif berisi 50.227 baris
dengan 35.161 train, 7.534 validation, dan 7.532 test. Angka metrik yang
masih tersimpan berasal dari run lama dengan 174 fitur; perbarui dengan
menjalankan Notebook 04–07 berurutan.

## Modul utama

| Modul | Tanggung jawab |
|---|---|
| src/parser/lin_parser.py | Parse LIN ke BoardRecord |
| src/parser/pbn_parser.py | Parse PBN ke BoardRecord |
| src/features/engineer.py | Membuat 164 fitur kanonik |
| src/features/dds.py | Membuat label dan fitur DDS untuk ekstraksi |
| src/preprocessing/dataset_builder.py | Label, deduplikasi, dan split deal-aware |
| src/models/ | Wrapper Random Forest, XGBoost, dan LightGBM |
| src/evaluation/metrics.py | Metrik klasifikasi biner |

## Seleksi dan evaluasi

Validation dipakai memilih model dan threshold. AP (Average Precision) menjadi
metrik utama karena fokus pada kelas positif. F1 macro, F1 positif,
precision/recall, accuracy, dan ROC-AUC menjadi metrik pendamping. Test hanya
mengestimasi kandidat yang telah dikunci; peringkat semua kandidat pada test
bersifat deskriptif.
