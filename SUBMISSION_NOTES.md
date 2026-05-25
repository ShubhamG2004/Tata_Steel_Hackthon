**Submission Notes — Defect Detection in Hot Rolling**

- **Short description:** Threshold-tuned gradient boosting classifier trained on `dataset/train.csv` to predict Alpha defect occurrence. Produces `expected_submission.csv` matching the sample submission format (`CoilID,Y`).
- **Files to upload:**
  - **Prediction file:** `expected_submission.csv` (CSV, 339 x 2)
  - **Source archive:** `submission.zip` (contains `src/`, `requirements.txt`, `README.md`, `expected_submission.csv`, and run scripts)

- **How to reproduce (commands to run in repository root):**

  Windows (PowerShell):
  ```powershell
  pip install -r requirements.txt
  python src\train.py --data-dir dataset --model-dir models
  python src\predict.py --model-dir models --data-dir dataset --output expected_submission.csv
  python src\package_submission.py
  ```

  Linux / macOS:
  ```bash
  pip install -r requirements.txt
  python3 src/train.py --data-dir dataset --model-dir models
  python3 src/predict.py --model-dir models --data-dir dataset --output expected_submission.csv
  python3 src/package_submission.py
  ```

- **Environment:** Python 3.8+; packages listed in `requirements.txt`.

- **Notes / compliance:**
  - No external datasets were used; only the provided `dataset/` files were consumed.
  - The `submission.zip` contains all source files required to reproduce the predictions and the produced `expected_submission.csv` used for the offline evaluation.
  - The produced `expected_submission.csv` has two columns: `CoilID` and `Y` (0/1). Ensure this file is uploaded as the prediction file on the problems page.

- **Evaluation details:**
  - Baseline model: threshold-tuned gradient boosting with sample weighting. You may further tune the model or threshold if needed.

- **Contact / notes for reviewers:**
  - To reproduce results or request logs, run the commands above. If you want me to run alternative thresholds or different models and produce a new `submission.zip`, tell me which changes to try.
