# A machine learning benchmark mapping the predictability frontier of Parkinson's disease progression in PPMI

This repository contains the analysis pipeline for the PPMI predictability-frontier
benchmark. It reproduces the full benchmark end to end from a PPMI data extract.

## Repository layout

```
.
├── configs/                 # Seeds, targets, regimes, model grids
├── data_preprocessing/      # Raw cleaning/pivot + per-task feature flattening
├── models/                  # Model definitions and hyperparameter grids
├── evaluation/              # Metrics, paired permutation tests, calibration
├── scripts/                 # SHAP, sensitivity, ablation, aggregation helpers
├── utils/                   # Plotting and IO helpers
├── run_all.py               # Master entry point
└── requirements.txt         # Pinned dependencies (Python 3.10)
```

## Reproduction

### 1. Environment

Python 3.10 is required.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data

PPMI data are restricted by a Data-Use Agreement and are **not** included in this
repository. The pipeline expects a local PPMI extract; set its location via
`RAW_DATA_DIR` in `configs/config.py`.

Access: <https://www.ppmi-info.org/access-data-specimens/download-data>

### 3. Build the processed feature tables

The static feature tables (demographics, genetics, CSF/plasma biomarkers, blood
chemistry) are built from the raw extract by the cleaning/pivot step. Run it once:

```bash
python data_preprocessing/01_data_cleaning_and_pivot.py
```

This writes `processed_data/` under your `RAW_DATA_DIR`, which the benchmark then
consumes. The longitudinal modalities are read directly from the raw extract by the
next step.

### 4. Run

```bash
python run_all.py
```

## License

MIT — see [`LICENSE`](LICENSE).

## Contact

Gianluca Leonardi · `gleonardi@fbk.eu` · Fondazione Bruno Kessler, Trento, Italy.
