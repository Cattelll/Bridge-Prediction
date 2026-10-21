# Eksperimen Robustness dan Validasi — 2026-10-21

Eksperimen ini menguji empat penguatan penelitian pada enam baseline DDS/noDDS dengan split penelitian yang sama. Jalankan dari root repo:

```powershell
.\.venv\Scripts\python.exe experiments/2026-10-21/run_robustness.py
```

Notebook ringkasan: `01_robustness_and_validation.ipynb`. Dataset utama penelitian sudah menggabungkan LIN dan PBN. Script dapat menjalankan evaluasi source-held-out internal dengan melatih model pada bagian LIN dan mengevaluasinya pada bagian PBN dari dataset gabungan. Model utama dan artefak penelitian yang sudah ada tidak ditimpa.

## Desain dan hasil penting

- **Ketidakpastian:** 500 bootstrap berkelompok per `_deal_key` pada test; baris dari deal yang sama tetap bersama. Label dapat berbeda dalam satu deal karena bidding/kontrak berbeda.
- **DDS vs noDDS:** perbandingan berpasangan pada test dengan interval bootstrap selisih dan exact sign test tingkat deal. Selisih accuracy DDS dibandingkan noDDS: RF **+5,32 pp** (95% CI **+4,21 sampai +6,41 pp**), XGBoost **+4,90 pp** (**+3,97 sampai +6,02 pp**), LightGBM **+7,75 pp** (**+6,39 sampai +9,08 pp**). P-value sign test tetap kecil setelah koreksi Holm.
- **Threshold:** threshold dipilih dengan memaksimalkan F1 macro di validation, grid 0,05 sampai 0,95. Pada test, F1 positif meningkat pada lima dari enam model; accuracy dapat berubah karena pertukaran antara recall positif dan false positives. Gunakan metrik test hanya sebagai estimasi akhir.
- **SHAP:** TreeExplainer default dihitung pada sampel validation sama berukuran 50 untuk keenam model. Porsi total mean |SHAP| dari fitur trik DDS di DDS: RF **28,47%**, XGBoost **30,52%**, LightGBM **31,55%**. Ini adalah kontribusi prediktif, bukan efek kausal; hasil sampel 50 bersifat eksploratif. Meski memakai TreeExplainer default pada semua model, SHAP tetap berada pada skala output masing-masing model; persentase antaralgoritma bukan uji beda formal.
- **Uji source-held-out internal:** latih hanya pada 12.471 baris LIN, pilih threshold dari validation LIN yang dikelompokkan menurut deal, latih ulang pada seluruh LIN, lalu evaluasi pada 37.756 baris PBN dari dataset gabungan. AP DDS: RF **0,564**, XGBoost **0,553**, LightGBM **0,573**; AP noDDS: **0,430**, **0,413**, **0,407**. Ini evaluasi internal lintas sumber, bukan validasi eksternal independen.

## Artefak

- `results/baseline_test_bootstrap_ci.csv`: metrik baseline dan CI 95% untuk accuracy, F1 macro, AP.
- `results/paired_dds_vs_nodds_inference.csv`: bootstrap selisih dan uji sign tingkat deal dengan koreksi Holm.
- `results/threshold_validation_test.csv`: threshold validation, confusion counts, dan metrik test pada threshold 0,5 serta threshold terpilih.
- `results/matched_exact_tree_shap.csv`: SHAP keenam model pada sampel validation yang sama.
- `results/source_heldout_lin_to_pbn.csv`: evaluasi source-held-out internal pada dataset gabungan.
- `results/metadata.json`: seed, pengaturan, dan batasan interpretasi.

## Batas interpretasi

Test bootstrap mengukur ketidakpastian akibat pengambilan sampel deal, bukan variasi akibat melatih ulang model dengan seed atau split lain. Uji source-held-out memeriksa transfer antarbagian sumber dalam dataset gabungan. Karena semua file berasal dari koleksi proyek, ini bukan validasi eksternal independen. SHAP tidak membuktikan sebab-akibat dan ringkasan global mean |SHAP| tidak menyatakan arah pengaruh fitur.
