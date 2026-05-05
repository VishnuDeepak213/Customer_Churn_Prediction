import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pandas as pd
import joblib
from src.model_training import ModelTrainer
from src.data_ingestion import DataIngestion
from src.preprocessing import preprocess_pipeline
from src.class_imbalance import ImbalanceHandler
from sklearn.model_selection import train_test_split


def main():
    ingestion = DataIngestion()
    df = pd.read_sql("SELECT * FROM raw_data.telco_churn", ingestion.engine)
    print(f"Data loaded: {df.shape}")

    X, y, preprocessor = preprocess_pipeline(df)
    print(f"Preprocessed: X={X.shape}, y={y.shape}")

    X_bal, y_bal = ImbalanceHandler.apply_smote(X, y)
    print(f"After SMOTE: X_bal={X_bal.shape}, y_bal={y_bal.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )
    print(f"Split: train={X_train.shape}, test={X_test.shape}")

    trainer = ModelTrainer()
    model, metrics = trainer.train_xgboost(X_train, y_train, X_test, y_test)

    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")

    models_dir = Path(__file__).resolve().parents[1] / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / 'best_model.pkl')
    joblib.dump(preprocessor, models_dir / 'preprocessor.pkl')

    print(f"Saved model to {models_dir / 'best_model.pkl'}")
    print(f"Saved preprocessor to {models_dir / 'preprocessor.pkl'}")
        
if __name__ == "__main__":
    main()