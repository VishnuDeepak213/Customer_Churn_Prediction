"""Phase 5 explainability driver.

Runs model evaluation plus SHAP and LIME analysis for the trained churn model.
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import DataIngestion
from src.preprocessing import preprocess_pipeline
from src.model_evaluation import ModelEvaluator
from src.shap_explanations import SHAPExplainer


def load_data():
    ingestion = DataIngestion()
    df = pd.read_sql("SELECT * FROM raw_data.telco_churn", ingestion.engine)
    X, y, _ = preprocess_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def main():
    print('=' * 80)
    print('PHASE 5: MODEL EXPLAINABILITY')
    print('=' * 80)

    model_path = PROJECT_ROOT / 'models' / 'best_model.pkl'
    print(f'Loading model from: {model_path}')
    model = joblib.load(model_path)

    print('Loading and preparing evaluation data...')
    X_train, X_test, y_train, y_test = load_data()
    print(f'Test set shape: {X_test.shape}')

    print('\n--- Evaluation ---')
    evaluator = ModelEvaluator(model, X_test, y_test)
    cm = evaluator.confusion_matrix_analysis()
    report = evaluator.classification_report_detailed()
    roc_auc, _, _, _ = evaluator.roc_auc_analysis()
    pr_auc, _, _ = evaluator.precision_recall_analysis()
    _, _ = evaluator.calibration_analysis()
    optimal_threshold, _, _ = evaluator.threshold_analysis()

    print('\nEvaluation summary:')
    print(f'Confusion matrix:\n{cm}')
    print(report)
    print(f'ROC-AUC: {roc_auc:.4f}')
    print(f'PR-AUC: {pr_auc:.4f}')
    print(f'Optimal threshold: {optimal_threshold:.4f}')

    print('\n--- SHAP ---')
    shap_sample = X_train.iloc[:1000]
    shap_explainer = SHAPExplainer(model, shap_sample)
    top_features = shap_explainer.top_features(n_features=10)
    print(top_features.to_string(index=False))

    print('\nPhase 5 explainability completed successfully.')


if __name__ == '__main__':
    main()
