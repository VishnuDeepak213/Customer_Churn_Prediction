import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation import ModelEvaluator
from src.shap_explanations import SHAPExplainer


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    alt = path.with_name('telco_churn.csv') if path.name == 'Telco_Churn.csv' else path.with_name('Telco_Churn.csv')
    if alt.exists():
        return pd.read_csv(alt)

    raise FileNotFoundError(f'CSV not found: {path}')


def main() -> None:
    project_root = PROJECT_ROOT
    reports_dir = project_root / 'reports' / 'phase5'
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = project_root / 'data' / 'raw' / 'Telco_Churn.csv'
    model_path = project_root / 'models' / 'best_model.pkl'
    preprocessor_path = project_root / 'models' / 'preprocessor.pkl'

    df = _read_csv_with_fallback(csv_path)
    if 'Churn' not in df.columns:
        raise KeyError('Expected Churn column in dataset')

    y = df['Churn'].map({'No': 0, 'Yes': 1})
    if y.isna().any():
        raise ValueError('Churn column contains values other than Yes/No')

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    X = preprocessor.transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int), test_size=0.2, stratify=y.astype(int), random_state=42
    )

    evaluator = ModelEvaluator(model, X_test, y_test)
    cm = evaluator.confusion_matrix_analysis(save_path=reports_dir / 'confusion_matrix.png')
    report_text = evaluator.classification_report_detailed()
    roc_auc, _, _, _ = evaluator.roc_auc_analysis(save_path=reports_dir / 'roc_curve.png')
    pr_auc, _, _ = evaluator.precision_recall_analysis(save_path=reports_dir / 'precision_recall_curve.png')
    evaluator.calibration_analysis(save_path=reports_dir / 'calibration_curve.png')

    shap_input = X_train.iloc[:200].astype(float)
    shap_explainer = SHAPExplainer(model, shap_input)
    shap_explainer.summary_plot(plot_type='bar', max_display=15, save_path=reports_dir / 'shap_summary_bar.png')
    try:
        shap_explainer.beeswarm_plot(max_display=15, save_path=reports_dir / 'shap_beeswarm.png')
    except Exception as exc:
        (reports_dir / 'shap_beeswarm_error.txt').write_text(str(exc), encoding='utf-8')
    shap_explainer.waterfall_plot(instance_idx=0, save_path=reports_dir / 'shap_waterfall_0.png')
    top_features = shap_explainer.top_features(n_features=15)
    top_features.to_csv(reports_dir / 'top_features.csv', index=False)

    target_auc = 0.85
    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'target_auc': target_auc,
        'roc_auc_meets_target': bool(roc_auc >= target_auc),
        'confusion_matrix': cm.tolist(),
    }
    (reports_dir / 'phase5_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (reports_dir / 'classification_report.txt').write_text(report_text, encoding='utf-8')

    print('PHASE5_ARTIFACTS_READY')
    print(f'reports_dir={reports_dir}')
    print(f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}")


if __name__ == '__main__':
    main()
