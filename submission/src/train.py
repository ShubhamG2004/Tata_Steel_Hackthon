import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main(data_dir="data", model_dir="models", random_state=42):
    os.makedirs(model_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")

    df = pd.read_csv(train_path)
    if 'CoilID' in df.columns:
        df = df.drop(columns=['CoilID'])

    if 'Y' not in df.columns:
        raise ValueError('Train file must contain target column `Y`.')

    X = df.drop(columns=['Y'])
    y = df['Y'].astype(int)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced')),
    ])

    pipeline.fit(X, y)

    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()
    main(data_dir=args.data_dir, model_dir=args.model_dir)
