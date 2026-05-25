#!/bin/bash
pip install -r requirements.txt
python3 src/train.py --data-dir dataset --model-dir models
python3 src/predict.py --model-dir models --data-dir dataset --output expected_submission.csv
python3 src/package_submission.py
