import argparse
import os
import joblib
import pandas as pd


def main(model_dir='models', data_dir='data', output='expected_submission.csv', threshold=0.5):
    model_path = os.path.join(model_dir, 'model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    test_path = os.path.join(data_dir, 'test.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    model = joblib.load(model_path)
    df = pd.read_csv(test_path)
    if 'CoilID' not in df.columns:
        raise ValueError('Test file must contain `CoilID` column.')

    ids = df['CoilID']
    X = df.drop(columns=['CoilID'])

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= float(threshold)).astype(int)

    out = pd.DataFrame({'CoilID': ids, 'Y': preds})
    out.to_csv(output, index=False)
    print(f"Wrote predictions to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output', default='expected_submission.csv')
    parser.add_argument('--threshold', default=0.5)
    args = parser.parse_args()
    main(model_dir=args.model_dir, data_dir=args.data_dir, output=args.output, threshold=args.threshold)
