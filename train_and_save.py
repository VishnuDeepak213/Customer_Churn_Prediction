import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_ingestion import DataIngestion
from src.preprocessing import preprocess_pipeline
from src.model_training import ModelTrainer
from src.class_imbalance import ImbalanceHandler

# Load raw data
print("Loading data...")
ingestion = DataIngestion()
df = pd.read_sql("SELECT * FROM raw_data.telco_churn", ingestion.engine)
print(f"✅ Loaded data: {df.shape}")

# Separate target
# Fit preprocessing pipeline and keep the fitted preprocessor
print("\nBuilding preprocessor...")
X_processed, y, preprocessor = preprocess_pipeline(df)
print(f"✅ Preprocessed data: {X_processed.shape}")

# Apply SMOTE for class balance
print("\nApplying SMOTE...")
X_bal, y_bal = ImbalanceHandler.apply_smote(X_processed, y)
print(f"✅ After SMOTE: {X_bal.shape}")

# Split data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# Train model
print("\nTraining XGBoost model...")
trainer = ModelTrainer()
model, metrics = trainer.train_xgboost(X_train, y_train, X_test, y_test)
print(f"✅ Model trained: AUC-ROC = {metrics['auc_roc']:.4f}")

# Save both artifacts
print("\nSaving artifacts...")
joblib.dump(model, 'models/best_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print("✅ Model and preprocessor saved!")
print("\n" + "="*60)
print("Ready to test API with Docker!")
print("="*60)
