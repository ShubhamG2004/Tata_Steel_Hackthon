import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


def build_pipeline(random_state=42, max_depth=4, learning_rate=0.05, max_iter=350, min_samples_leaf=15):
    model = HistGradientBoostingClassifier(
        random_state=random_state,
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=0.1,
        early_stopping=False,
    )
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', model),
    ])


def select_threshold(y_true, y_prob):
    positive_probs = y_prob[np.asarray(y_true) == 1]
    if positive_probs.size == 0:
        return 0.5, 0.0, 0.0

    threshold = float(np.min(positive_probs))
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return threshold, float(precision), float(recall)


def fit_with_validation(X, y, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    train_sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

    candidate_configs = [
        {'max_depth': 3, 'learning_rate': 0.05, 'max_iter': 300, 'min_samples_leaf': 20},
        {'max_depth': 4, 'learning_rate': 0.05, 'max_iter': 350, 'min_samples_leaf': 15},
        {'max_depth': 5, 'learning_rate': 0.03, 'max_iter': 450, 'min_samples_leaf': 10},
        {'max_depth': 6, 'learning_rate': 0.05, 'max_iter': 300, 'min_samples_leaf': 10},
    ]

    best_result = None
    for config in candidate_configs:
        pipeline = build_pipeline(random_state=random_state, **config)
        pipeline.fit(X_train, y_train, clf__sample_weight=train_sample_weight)
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        threshold, precision, recall = select_threshold(y_val, val_prob)
        score = (precision, recall, threshold)

        if best_result is None or score > best_result['score']:
            best_result = {
                'pipeline': pipeline,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'config': config,
                'score': score,
            }

    return best_result


def main(data_dir="dataset", model_dir="models", random_state=42):
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

    tuned = fit_with_validation(X, y, random_state=random_state)

    full_sample_weight = compute_sample_weight(class_weight='balanced', y=y)
    final_pipeline = build_pipeline(random_state=random_state, **tuned['config'])
    final_pipeline.fit(X, y, clf__sample_weight=full_sample_weight)

    model_path = os.path.join(model_dir, 'model.joblib')
    artifact = {
        'pipeline': final_pipeline,
        'threshold': tuned['threshold'],
        'feature_columns': list(X.columns),
        'validation_precision': tuned['precision'],
        'validation_recall': tuned['recall'],
        'model_config': tuned['config'],
    }
    joblib.dump(artifact, model_path)
    print(f"Saved trained model to {model_path}")
    print(
        "Validation precision={precision:.4f}, recall={recall:.4f}, threshold={threshold:.6f}".format(
            precision=tuned['precision'],
            recall=tuned['recall'],
            threshold=tuned['threshold'],
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='dataset')
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()
    main(data_dir=args.data_dir, model_dir=args.model_dir)
