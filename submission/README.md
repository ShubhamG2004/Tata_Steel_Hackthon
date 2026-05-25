# Defect Detection in Hot Rolling — Reproducible Baseline

This repository provides a minimal, reproducible baseline for the "Defect Detection in Hot Rolling" challenge. It trains a Random Forest on `data/train.csv` and produces a submission file `expected_submission.csv` for the test set `data/test.csv`.

Structure:
- `data/` — place `train.csv`, `test.csv`, and (optional) `sample_submission.csv` here.
- `src/` — training, prediction, and packaging scripts.
- `models/` — trained model artifact created after running training.
- `expected_submission.csv` — output prediction file after running the pipeline.

Quick start (after placing data in `data/`):

Windows (PowerShell):
```powershell
pip install -r requirements.txt
python src\train.py --data-dir data --model-dir models
python src\predict.py --model-dir models --data-dir data --output expected_submission.csv --threshold 0.5
python src\package_submission.py
```

Linux / macOS:
```bash
pip install -r requirements.txt
python3 src/train.py --data-dir data --model-dir models
python3 src/predict.py --model-dir models --data-dir data --output expected_submission.csv --threshold 0.5
python3 src/package_submission.py
```

Notes:
- Do NOT include external datasets in the submission archive.
- The `expected_submission.csv` must match the sample submission format (columns: `CoilID`, `Y`) and have 339 rows for the provided test split.
