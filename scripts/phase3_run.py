"""
Phase 3: Feature Engineering Pipeline
Orchestrates: Data Loading → Preprocessing → Feature Engineering → Class Imbalance Handling
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion import DataIngestion
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import FeatureEngineer
from src.class_imbalance import ImbalanceHandler

print("=" * 80)
print("PHASE 3: FEATURE ENGINEERING PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA FROM DATABASE
# ============================================================================
print("\n[STEP 1] Loading data from raw_data.telco_churn...")
ingestion = DataIngestion()
df_raw = pd.read_sql("SELECT * FROM raw_data.telco_churn", ingestion.engine)

print(f"✅ Original shape: {df_raw.shape}")
print(f"✅ Target distribution:\n{df_raw['Churn'].value_counts()}")
print(f"✅ Missing values: {df_raw.isnull().sum().sum()}")

# ============================================================================
# STEP 2: PREPROCESSING (Handle missing values, encode, scale)
# ============================================================================
print("\n[STEP 2] Applying preprocessing (missing values, encoding, scaling)...")
X, y, processor = preprocess_pipeline(df_raw)

print(f"✅ After preprocessing shape: {X.shape}")
print(f"✅ Missing values after preprocessing: {X.isnull().sum().sum()}")
print(f"✅ Data types:\n{X.dtypes.value_counts()}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING (Create engineered features)
# ============================================================================
print("\n[STEP 3] Applying feature engineering...")
X_engineered = FeatureEngineer.engineer_all_features(df_raw)

new_features = [col for col in X_engineered.columns if col not in df_raw.columns]
num_new = len(new_features)

print(f"✅ Original features: {df_raw.shape[1]}")
print(f"✅ New features created: {num_new}")
print(f"✅ Feature list: {new_features}")
print(f"✅ Engineered dataset shape: {X_engineered.shape}")

# ============================================================================
# STEP 4: CLASS IMBALANCE HANDLING (Apply SMOTE)
# ============================================================================
print("\n[STEP 4] Applying class imbalance handling (SMOTE)...")
print(f"📊 Class distribution BEFORE SMOTE:")
print(f"   No Churn: {(y == 0).sum()}")
print(f"   Churn: {(y == 1).sum()}")
print(f"   Ratio: {(y == 1).sum() / (y == 0).sum():.2%}")

X_balanced, y_balanced = ImbalanceHandler.apply_smote(X, y, sampling_strategy=0.5)

print(f"\n📊 Class distribution AFTER SMOTE:")
print(f"   No Churn: {(y_balanced == 0).sum()}")
print(f"   Churn: {(y_balanced == 1).sum()}")
print(f"   Ratio: {(y_balanced == 1).sum() / (y_balanced == 0).sum():.2%}")
print(f"✅ Balanced dataset shape: {X_balanced.shape}")

# ============================================================================
# STEP 5: SAVE PROCESSED DATA
# ============================================================================
print("\n[STEP 5] Saving processed data...")

# Combine features and target
processed_df = pd.concat([X_balanced, pd.Series(y_balanced, name='Churn')], axis=1)

# Save to PostgreSQL
processed_df.to_sql(
    'telco_churn_processed',
    ingestion.engine,
    schema='processed_data',
    if_exists='replace',
    index=False
)
print(f"✅ Saved to PostgreSQL: processed_data.telco_churn_processed ({processed_df.shape[0]} rows, {processed_df.shape[1]} cols)")

# Save to CSV
output_dir = Path(__file__).parent.parent / 'data' / 'processed'
output_dir.mkdir(parents=True, exist_ok=True)

features_path = output_dir / 'phase3_features.csv'
target_path = output_dir / 'phase3_target.csv'

X_balanced.to_csv(features_path, index=False)
pd.Series(y_balanced, name='Churn').to_csv(target_path, index=False)

print(f"✅ Saved features: {features_path}")
print(f"✅ Saved target: {target_path}")

# ============================================================================
# VALIDATION SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3 COMPLETION SUMMARY")
print("=" * 80)
print(f"✅ Original dataset: {df_raw.shape}")
print(f"✅ After preprocessing: {X.shape}")
print(f"✅ After feature engineering: {X_engineered.shape} (added {num_new} features)")
print(f"✅ After SMOTE balancing: {X_balanced.shape}")
print(f"✅ Class balance: {(y_balanced == 1).sum() / (y_balanced == 0).sum():.2%}")
print(f"✅ Saved to: PostgreSQL, CSV")
print("=" * 80)
print("✨ Phase 3 pipeline complete! Ready for Phase 4 (Model Training)")
print("=" * 80)