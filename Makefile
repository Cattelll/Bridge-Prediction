# Bridge Contract Prediction - Makefile
# Gunakan dari Git Bash / WSL: make <target>
# Pastikan Python 3.12 tersedia di PATH

PYTHON := py -3.12
NB     := jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=bridge

.PHONY: help setup build train report notebooks experiments all clean validate

# -- Default -----------------------------------------------------------
help:
	@echo ""
	@echo "  Bridge Contract Prediction - Automation"
	@echo "  ----------------------------------------"
	@echo "  make setup       Install dependencies + daftarkan kernel Jupyter"
	@echo "  make build       Build dataset dari file LIN (data/raw/)"
	@echo "  make train       Latih RF, XGBoost, LightGBM + evaluasi"
	@echo "  make report      Generate REPORT.md dari hasil tersimpan"
	@echo "  make notebooks   Jalankan semua notebook 01-04 secara berurutan"
	@echo "  make experiments Jalankan notebook 05 (class imbalance)"
	@echo "  make all         build + train + report (pipeline lengkap)"
	@echo "  make clean       Hapus data/processed/ dan outputs/"
	@echo ""

# -- Setup -------------------------------------------------------------
setup:
	$(PYTHON) -m pip install -e ".[notebook,experiments,dev]"
	$(PYTHON) -m ipykernel install --user --name bridge --display-name "Python 3.12 (Bridge)"
	@echo "Setup selesai. Kernel 'Python 3.12 (Bridge)' terdaftar."

# -- Dataset build -----------------------------------------------------
build:
	@echo "[BUILD] Membangun dataset dari data/raw/ ..."
	$(PYTHON) -c "\
import sys; sys.path.insert(0,'.');\
from src.preprocessing import build_dataset;\
build_dataset('data/raw', 'data/processed', target_col='target_base',\
              train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)"
	@echo "[BUILD] Selesai. Dataset tersimpan di data/processed/"

# -- Training + evaluasi -----------------------------------------------
train:
	$(PYTHON) scripts/run_pipeline.py

# -- Report ------------------------------------------------------------
report:
	$(PYTHON) scripts/report.py

# -- Notebooks ---------------------------------------------------------
notebooks:
	$(NB) notebooks/01_dataset_analysis.ipynb
	$(NB) notebooks/02_eda_features.ipynb
	$(NB) notebooks/03_modeling.ipynb
	$(NB) notebooks/04_evaluation.ipynb
	@echo "[NOTEBOOKS] Notebook 01-04 selesai dieksekusi."

# -- Experiments -------------------------------------------------------
experiments:
	$(NB) notebooks/05_class_imbalance.ipynb
	@echo "[EXPERIMENTS] Notebook 05 selesai dieksekusi."

# -- Semua sekaligus ---------------------------------------------------
all: build train report

# -- Bersihkan data & output -------------------------------------------
clean:
	@echo "[CLEAN] Menghapus data/processed/ ..."
	-rm -f data/processed/*.csv data/processed/*.pkl data/processed/*.json
	@echo "[CLEAN] Menghapus outputs/ ..."
	-rm -f outputs/models/*.pkl
	-rm -f outputs/results/*.png outputs/results/*.json outputs/results/*.csv
	@echo "[CLEAN] Selesai."

# -- Validasi ----------------------------------------------------------
validate:
	$(PYTHON) scripts/validate_parser.py
	$(PYTHON) scripts/validate_features.py
