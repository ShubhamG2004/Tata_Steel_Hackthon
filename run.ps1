pip install -r requirements.txt
python src\train.py --data-dir dataset --model-dir models
python src\predict.py --model-dir models --data-dir dataset --output expected_submission.csv
python src\package_submission.py
