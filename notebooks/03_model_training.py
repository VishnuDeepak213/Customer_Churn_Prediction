"""Phase 4 model training workflow in .py form.

This is the notebook-to-script version of the Phase 4 training steps from
PHASE_4_MODEL_TRAINING.md. It runs locally without needing an .ipynb file.

Flow:
1. Load Phase 3 processed data
2. Split into train/test
3. Train baseline models
4. Train advanced models
5. Optionally run Optuna tuning for XGBoost
6. Save the best model and preprocessor artifact
"""

from __future__ import annotations

from pathlib import Path
import json
from pyexpat import model
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib
from src.preprocessing import build_preprocessor


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_ingestion import DataIngestion
from src.preprocessing import preprocess_pipeline
from src.class_imbalance import ImbalanceHandler

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH = MODELS_DIR / "phase4_metrics.json"


class ModelTrainer:
    def __init__(self):
        self.models = {}

    @staticmethod
    def _metrics(y_true, y_pred, y_pred_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = self._metrics(y_test, y_pred, y_pred_proba)
        self.models["logistic"] = model
        return model, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = self._metrics(y_test, y_pred, y_pred_proba)
        self.models["random_forest"] = model
        return model, metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = self._metrics(y_test, y_pred, y_pred_proba)
        self.models["xgboost"] = model
        return model, metrics

    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = self._metrics(y_test, y_pred, y_pred_proba)
        self.models["lightgbm"] = model
        return model, metrics
    # Build and fit preprocessor on raw data
    preprocessor = build_preprocessor(X_processed)  # X_raw = original data before any preprocessing
    X_processed = preprocessor.fit_transform(X_raw)

    # Train model on processed data
    model.fit(X_processed, y)

    # Save both
    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    print("✅ Model and preprocessor saved!")


class HyperparameterTuner:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.study = None

    def objective_xgboost(self, trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }

        model = xgb.XGBClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=float((self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1)),
        )
        model.fit(self.X_train, self.y_train)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        return roc_auc_score(self.y_test, y_pred_proba)

    def tune_xgboost(self, n_trials=50, timeout=None):
        if optuna is None:
            raise ImportError("optuna is not installed. Install it or skip tuning.")

        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        self.study.optimize(self.objective_xgboost, n_trials=n_trials, timeout=timeout)
        return self.study.best_params

    def train_best_xgboost(self):
        if self.study is None:
            raise ValueError("Run tune_xgboost() first")

        best_params = self.study.best_params.copy()
        model = xgb.XGBClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=float((self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1)),
        )
        model.fit(self.X_train, self.y_train)
        return model


def save_metrics(metrics_by_model):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics_by_model, handle, indent=2)


def print_results(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")


def main():
    print("=" * 80)
    print("PHASE 4: MODEL TRAINING")
    print("=" * 80)

    # Cell 1: Setup & Load Data
    ingestion = DataIngestion()
    df_raw = pd.read_sql("SELECT * FROM raw_data.telco_churn", ingestion.engine)
    print(f"Loaded raw data: {df_raw.shape}")

    X, y, processor = preprocess_pipeline(df_raw)
    print(f"Preprocessed data: X={X.shape}, y={y.shape}")

    X_balanced, y_balanced = ImbalanceHandler.apply_smote(X, y, sampling_strategy=0.5)
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced,
        y_balanced,
        test_size=0.2,
        stratify=y_balanced,
        random_state=42,
    )

    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Train target distribution:\n{pd.Series(y_train).value_counts()}")

    trainer = ModelTrainer()
    all_metrics = {}

    # Cell 2: Train Baseline Models
    lr_model, lr_metrics = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    all_metrics["LogisticRegression"] = lr_metrics
    all_metrics["RandomForest"] = rf_metrics

    print_results("Baseline Models", lr_metrics)
    print_results("Random Forest", rf_metrics)

    # Cell 3: Train Advanced Models
    xgb_model, xgb_metrics = trainer.train_xgboost(X_train, y_train, X_test, y_test)
    lgb_model, lgb_metrics = trainer.train_lightgbm(X_train, y_train, X_test, y_test)
    all_metrics["XGBoost"] = xgb_metrics
    all_metrics["LightGBM"] = lgb_metrics

    print_results("XGBoost", xgb_metrics)
    print_results("LightGBM", lgb_metrics)

    # Cell 4: Hyperparameter Tuning (Optional)
    best_params = None
    best_xgb = None
    best_xgb_metrics = None

    if optuna is not None:
        tuner = HyperparameterTuner(X_train, y_train, X_test, y_test)
        # Reduce trials for a practical local run; increase if you want a deeper search.
        best_params = tuner.tune_xgboost(n_trials=25, timeout=1800)
        best_xgb = tuner.train_best_xgboost()
        y_pred = best_xgb.predict(X_test)
        y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
        best_xgb_metrics = trainer._metrics(y_test, y_pred, y_pred_proba)
        all_metrics["TunedXGBoost"] = best_xgb_metrics
        print_results("Tuned XGBoost", best_xgb_metrics)
    else:
        print("Optuna not installed, skipping tuning step.")

    # Cell 5: Cross-Validation & Model Comparison
    best_model_name = max(all_metrics, key=lambda name: all_metrics[name]["auc_roc"])
    best_model_metrics = all_metrics[best_model_name]
    print(f"\nBest model: {best_model_name}")
    print(f"AUC-ROC: {best_model_metrics['auc_roc']:.4f}")
    print(f"F1-Score: {best_model_metrics['f1']:.4f}")
    print(f"Precision: {best_model_metrics['precision']:.4f}")
    print(f"Recall: {best_model_metrics['recall']:.4f}")

    # Cell 6: Save Best Model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if best_model_name == "TunedXGBoost" and best_xgb is not None:
        model_to_save = best_xgb
    elif best_model_name == "XGBoost":
        model_to_save = xgb_model
    elif best_model_name == "LightGBM":
        model_to_save = lgb_model
    elif best_model_name == "RandomForest":
        model_to_save = rf_model
    else:
        model_to_save = lr_model

    joblib.dump(model_to_save, BEST_MODEL_PATH)
    joblib.dump(processor, PREPROCESSOR_PATH)
    save_metrics(all_metrics)

    print(f"\n✅ Best model saved to {BEST_MODEL_PATH}")
    print(f"✅ Preprocessor saved to {PREPROCESSOR_PATH}")
    print(f"✅ Metrics saved to {METRICS_PATH}")
    print("\nPhase 4 model training complete.")


if __name__ == "__main__":
    main()