"""Phase 4 model training script.

This local script replaces the notebook workflow for Step 3 and does:
- load Phase 3 processed features/target
- split into train/holdout sets
- train an XGBoost classifier
- evaluate on the holdout set
- save the model artifact
"""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

FEATURES_PATH = DATA_DIR / 'phase3_features.csv'
TARGET_PATH = DATA_DIR / 'phase3_target.csv'
MODEL_PATH = MODELS_DIR / 'best_model.pkl'
METRICS_PATH = MODELS_DIR / 'phase4_metrics.json'


def load_data():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f'Missing features file: {FEATURES_PATH}')
    if not TARGET_PATH.exists():
        raise FileNotFoundError(f'Missing target file: {TARGET_PATH}')

    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(TARGET_PATH).iloc[:, 0]

    if y.name != 'Churn':
        y.name = 'Churn'

    return X, y


def train_model(X_train, y_train):
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics


def main():
    print('=' * 80)
    print('PHASE 4: MODEL TRAINING')
    print('=' * 80)

    X, y = load_data()
    print(f'Loaded processed data: X={X.shape}, y={y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print(f'Train split: X_train={X_train.shape}, X_test={X_test.shape}')
    print(f'Train target distribution:\n{y_train.value_counts()}')
    print(f'Test target distribution:\n{y_test.value_counts()}')

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    print('\nModel metrics')
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print('Confusion matrix:')
    print(metrics['confusion_matrix'])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f'\nSaved model to: {MODEL_PATH}')

    with METRICS_PATH.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    print(f'Saved metrics to: {METRICS_PATH}')

    split_dir = DATA_DIR
    X_train.to_csv(split_dir / 'phase4_train_features.csv', index=False)
    X_test.to_csv(split_dir / 'phase4_test_features.csv', index=False)
    y_train.to_csv(split_dir / 'phase4_train_target.csv', index=False)
    y_test.to_csv(split_dir / 'phase4_test_target.csv', index=False)
    print('Saved train/test split CSV files in data/processed/')

    print('\nPhase 4 model training complete.')


if __name__ == '__main__':
    main()
