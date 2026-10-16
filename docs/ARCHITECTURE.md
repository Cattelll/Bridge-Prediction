# Arsitektur dan Alur Pipeline

Pipeline resmi membangun klasifikasi biner kecocokan kontrak par dari rekaman
LIN dan PBN. Definisi label dan skema fitur berada di `src/`; notebook
`01`?`06` mengorkestrasi ekstraksi, eksplorasi, pelatihan, dan evaluasi.

## Alur data

```text
data/raw/*.lin + data/raw_pbn/*.pbn
  -> LINParser / PBNParser -> BoardRecord
  -> 164 fitur kanonik + metadata deal dan target kontrak aktual
  -> DDS v4: 10 fitur trik DD + kontrak par untuk membuat label biner
  -> buang kolom par yang langsung membentuk label
  -> deduplikasi pasangan fitur-label yang identik
  -> StratifiedGroupKFold, grup = tanda tangan deal 52 kartu
  -> data/processed/{train,val,test,full}.csv
  -> RandomForest / XGBoost / LightGBM
  -> validation untuk pemilihan konfigurasi; test untuk estimasi akhir
```

Label `matches_par_contract` bernilai 1 jika kontrak aktual cocok dengan salah
satu kontrak par DDS pada level, strain, status double/redouble, dan sisi
declarer (NS/EW). Cache v4 wajib karena menyimpan semua kontrak par yang
memiliki skor sama. DDS juga menghasilkan 10 fitur jumlah trik (NS dan EW untuk
masing-masing strain); fitur ini tetap dapat dipakai setelah kolom par yang
menurunkan label dikeluarkan.

## Pencegahan kebocoran split

Setiap deal diberi `_deal_key` dari kartu pada keempat tangan dalam urutan
kanonik. Identitas ini tidak bergantung pada nama file, nomor papan, format
LIN/PBN, atau urutan kartu dalam file. `StratifiedGroupKFold` membagi berdasarkan
`_deal_key`, sehingga salinan deal dari sumber berbeda tetap berada pada satu
partisi. Builder memeriksa overlap key antar train, validation, dan test sebelum
menyimpan CSV. `_deal_key` hanya metadata dan tidak pernah menjadi fitur model.

Pengulangan dengan fitur **dan label** yang sama dihapus. Baris dengan fitur
ringkas sama tetapi label berbeda dipertahankan sebagai observasi berbeda.

## Modul utama

| Modul | Tanggung jawab |
|---|---|
| `src/parser/lin_parser.py` | Parse LIN, kartu, auction, kontrak dan metadata board |
| `src/parser/pbn_parser.py` | Parse PBN ke bentuk `BoardRecord` yang sama |
| `src/features/engineer.py` | Membuat 164 fitur kanonik dan kolom target historis |
| `src/features/dds.py` | Menghitung/membaca fitur DDS dan kontrak par |
| `src/preprocessing/dataset_builder.py` | Membuat label, menghapus kolom leakage, deduplikasi dan split grup deal |
| `src/models/` | Wrapper Random Forest, XGBoost, dan LightGBM |
| `src/evaluation/metrics.py` | Accuracy, precision/recall/F1, ROC-AUC, dan Average Precision |
| `scripts/run_pipeline.py` | Membangun dataset biner LIN+PBN dan melatih baseline dari config |

## Evaluasi

Validation dipakai memilih model dan threshold. Test dipakai sekali untuk
melaporkan performa pilihan yang telah dikunci. Tabel beberapa kandidat pada
test bersifat deskriptif dan tidak boleh dipakai untuk memilih pemenang.

Untuk fokus kelas positif, laporkan Average Precision (AP; disebut PR-AUC di
sebagian plot), ROC-AUC sebagai metrik ranking tambahan, serta F1/precision/recall
pada threshold yang dipilih di validation. Angka proporsi kelas dan rasio
mayoritas:minoritas perlu dilaporkan bersama metrik.

## Perintah pipeline

```powershell
python scripts/build_dds_cache.py  # hanya jika cache v4 belum tersedia
python scripts/run_pipeline.py    # membangun split deal-aware dan melatih baseline
```

Notebook 01?06 menyediakan EDA dan eksperimen tambahan. `scripts/run_pipeline.py`
menggunakan target biner dan cache v4; jangan memakai pipeline multiclass lama
dari eksperimen arsip.
