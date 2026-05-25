import argparse
import os

import joblib
import pandas as pd


def main(model_dir='models', data_dir='dataset', output='expected_submission.csv', threshold=None):
    model_path = os.path.join(model_dir, 'model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    test_path = os.path.join(data_dir, 'test.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        model = artifact.get('pipeline')
        stored_threshold = artifact.get('threshold', 0.5)
        feature_columns = artifact.get('feature_columns')
    else:
        model = artifact
        stored_threshold = 0.5
        feature_columns = None

    df = pd.read_csv(test_path)
    if 'CoilID' not in df.columns:
        raise ValueError('Test file must contain `CoilID` column.')

    ids = df['CoilID']
    X = df.drop(columns=['CoilID'])
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns)

    probs = model.predict_proba(X)[:, 1]
    effective_threshold = stored_threshold if threshold is None else float(threshold)
    preds = (probs >= effective_threshold).astype(int)

    out = pd.DataFrame({'CoilID': ids, 'Y': preds})
    out.to_csv(output, index=False)
    print(f"Wrote predictions to {output} using threshold {effective_threshold:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--data-dir', default='dataset')
    parser.add_argument('--output', default='expected_submission.csv')
    parser.add_argument('--threshold', type=float, default=None)
    args = parser.parse_args()
    main(model_dir=args.model_dir, data_dir=args.data_dir, output=args.output, threshold=args.threshold)
