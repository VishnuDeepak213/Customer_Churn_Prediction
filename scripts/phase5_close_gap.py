from pathlib import Path
import json
import sys

import joblib
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_pipeline


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    alt = path.with_name('telco_churn.csv') if path.name == 'Telco_Churn.csv' else path.with_name('Telco_Churn.csv')
    if alt.exists():
        return pd.read_csv(alt)
    raise FileNotFoundError(f'CSV not found: {path}')


def main() -> None:
    data_path = PROJECT_ROOT / 'data' / 'raw' / 'Telco_Churn.csv'
    df = _read_csv_with_fallback(data_path)

    X, y, preprocessor = preprocess_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'auc',
            'verbosity': 0,
            'tree_method': 'hist',
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        return float(auc)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=40, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc',
        'verbosity': 0,
        'tree_method': 'hist',
    })

    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_proba))

    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / 'best_model.pkl')
    joblib.dump(preprocessor, models_dir / 'preprocessor.pkl')

    reports_dir = PROJECT_ROOT / 'reports' / 'phase5'
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'optimized_auc': test_auc,
        'target_auc': 0.89,
        'meets_target': bool(test_auc > 0.89),
        'best_params': study.best_params,
    }
    (reports_dir / 'phase5_auc_optimization.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(f'BEST_AUC={test_auc:.6f}')
    print(f"MEETS_TARGET={payload['meets_target']}")


if __name__ == '__main__':
    main()
