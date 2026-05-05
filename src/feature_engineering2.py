import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from src.data_ingestion import DataIngestion
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import FeatureEngineer
from src.class_imbalance import ImbalanceHandler

# Load data
ingestion = DataIngestion()
df = pd.read_sql("SELECT * FROM raw_data.telco_churn", ingestion.engine)

print(f"Original shape: {df.shape}")
print(f"Target distribution:\\n{df['Churn'].value_counts()}")

### Cell 2: Apply Preprocessing

# Preprocess (handle missing, encode, scale)
X, y, processor = preprocess_pipeline(df)

print(f"\\nAfter preprocessing:\\n{X.shape}")
print(f"Feature types:\\n{X.dtypes.value_counts()}")


### Cell 3: Apply Feature Engineering

# Add engineered features
X_engineered = FeatureEngineer.engineer_all_features(df)

print(f"\\nNew features created: {X_engineered.shape[1] - df.shape[1]}")
print(f"\\nNew features:")
new_features = [col for col in X_engineered.columns if col not in df.columns]
print(new_features)


### Cell 4: Handle Class Imbalance

# Apply SMOTE
X_balanced, y_balanced = ImbalanceHandler.apply_smote(X, y, sampling_strategy=0.5)

print(f"\\nBalanced dataset shape: {X_balanced.shape}")
print(f"Class distribution after balancing:\\n{pd.Series(y_balanced).value_counts()}")


### Cell 5: Save Preprocessed Data

# Save to processed_data schema
processed_df = pd.concat([X_balanced, pd.Series(y_balanced, name='Churn')], axis=1)

# Upload to database
processed_df.to_sql(
    'telco_churn_processed',
    ingestion.engine,
    schema='processed_data',
    if_exists='replace',
    index=False
)

print(f"✅ Saved processed data to PostgreSQL")