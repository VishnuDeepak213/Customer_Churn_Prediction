import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
    
    def handle_missing_values(self, df, numerical_cols, categorical_cols):
        """Handle missing values in dataset"""
        df_clean = df.copy()
        
        # Numerical columns: median imputation
        if numerical_cols:
            df_clean[numerical_cols] = self.imputer_num.fit_transform(df_clean[numerical_cols])
            logger.info(f"✅ Imputed {len(numerical_cols)} numerical columns with median")
        
        # Categorical columns: mode imputation
        if categorical_cols:
            df_clean[categorical_cols] = self.imputer_cat.fit_transform(df_clean[categorical_cols])
            logger.info(f"✅ Imputed {len(categorical_cols)} categorical columns with mode")
        
        return df_clean
    
    def remove_outliers_iqr(self, df, numerical_cols, multiplier=1.5):
        """Remove outliers using IQR method"""
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        for col in numerical_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Keep only rows within bounds
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed = initial_rows - len(df_clean)
        logger.info(f"✅ Removed {removed} outlier rows using IQR method")
        return df_clean
    
    def encode_categorical(self, df, categorical_cols, fit=True):
        """Encode categorical variables
        Options:
        - Binary: LabelEncoder (0, 1)
        - Multi-class: OneHotEncoder (one column per category)
        """
        df_encoded = df.copy()
        
        for col in categorical_cols:
            unique_values = df_encoded[col].nunique()
            
            if unique_values == 2:  # Binary encoding
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
                logger.info(f"✅ Binary encoded '{col}': {df_encoded[col].unique()}")
            
            else:  # Multi-class (keep for one-hot later)
                pass
        
        return df_encoded
    
    def one_hot_encode(self, df, categorical_cols):
        """One-hot encode multi-class categorical variables"""
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logger.info(f"✅ One-hot encoded {len(categorical_cols)} columns")
        logger.info(f"Result shape: {df_encoded.shape}")
        return df_encoded
    
    def scale_numerical_features(self, df, numerical_cols, fit=True):
        """Scale numerical features to zero mean, unit variance"""
        df_scaled = df.copy()
        
        if fit:
            df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
        else:
            df_scaled[numerical_cols] = self.scaler.transform(df_scaled[numerical_cols])
        
        logger.info(f"✅ Scaled {len(numerical_cols)} numerical features")
        return df_scaled
    
    def get_feature_stats_after_preprocessing(self, df):
        """Print stats after preprocessing"""
        print("\\n" + "="*60)
        print(f"PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Shape: {df.shape}")
        print(f"\nMissing values:\\n{df.isnull().sum().sum()}")
        print(f"\nData types:\\n{df.dtypes.value_counts()}")
        print(f"\nNumerical features: {df.select_dtypes(include=[np.number]).shape[1]}")
        print("="*60 + "\\n")

# Usage example in notebook
def preprocess_pipeline(df, target_col='Churn'):
    """Complete preprocessing pipeline

    Steps implemented to match Phase 3 spec:
    - Map target to 0/1 (`No` -> 0, `Yes` -> 1)
    - Drop `customerID` where present
    - Median imputation for numerical, mode for categorical
    - Binary label-encoding for 2-valued categoricals
    - One-hot encode multi-class categoricals (no drop_first)
    - Scale original numerical columns with StandardScaler
    """
    processor = DataPreprocessor()

    # Map target and separate
    if target_col not in df.columns:
        raise KeyError(f"Expected target column `{target_col}` in dataframe")

    y = df[target_col].map({'No': 0, 'Yes': 1})
    if y.isna().any():
        # fallback: keep original values if mapping not applicable
        y = df[target_col]

    X = df.drop(columns=[target_col])

    # Drop customerID if present (not a feature)
    if 'customerID' in X.columns:
        X = X.drop(columns=['customerID'])

    # Identify column types before encoding
    # Convert object columns that are mostly numeric to numeric
    obj_cols = X.select_dtypes(include='object').columns.tolist()
    coerced_numeric = []
    for col in obj_cols:
        coerced = pd.to_numeric(X[col], errors='coerce')
        numeric_ratio = coerced.notna().mean()
        if numeric_ratio >= 0.8:
            # treat as numeric
            X[col] = coerced.fillna(coerced.median())
            coerced_numeric.append(col)

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.select_dtypes(include='object').columns.tolist()]

    logger.info(f"Numerical columns: {numerical_cols}")
    logger.info(f"Categorical columns: {categorical_cols}")

    # Step 1: Handle missing values
    X = processor.handle_missing_values(X, numerical_cols, categorical_cols)

    # Step 2: Binary encode 2-valued categoricals
    binary_cols = [c for c in categorical_cols if X[c].nunique() == 2]
    multi_cols = [c for c in categorical_cols if X[c].nunique() > 2]

    if binary_cols:
        X = processor.encode_categorical(X, binary_cols, fit=True)

    # Step 3: One-hot encode multi-class categoricals (preserve all columns)
    if multi_cols:
        X = processor.one_hot_encode(X, multi_cols)

    # Step 4: Scale original numerical features
    if numerical_cols:
        X = processor.scale_numerical_features(X, numerical_cols, fit=True)

    # Step 5: Print stats
    processor.get_feature_stats_after_preprocessing(X)

    return X, y.astype(int), processor

def build_preprocessor(X):
    """Build sklearn ColumnTransformer for API inference."""
    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ],
        remainder='passthrough'
    )
    return preprocessor


class TabularPreprocessor:
    """Fit and reuse a simple tabular preprocessing contract for the churn model."""

    def __init__(self, target_col='Churn', drop_columns=None):
        self.target_col = target_col
        self.drop_columns = list(drop_columns or ['customerID'])
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_defaults = {}
        self.categorical_defaults = {}
        self.fitted = False

    def _prepare_features(self, df):
        X = df.copy()

        if self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])

        drop_existing = [column for column in self.drop_columns if column in X.columns]
        if drop_existing:
            X = X.drop(columns=drop_existing)

        return X

    def fit(self, df):
        X = self._prepare_features(df)

        numeric_frames = {}

        for column in X.columns:
            coerced = pd.to_numeric(X[column], errors='coerce')
            numeric_ratio = coerced.notna().mean()

            if numeric_ratio >= 0.8:
                self.numeric_cols.append(column)
                median_value = float(coerced.median()) if coerced.notna().any() else 0.0
                if np.isnan(median_value):
                    median_value = 0.0
                self.numeric_defaults[column] = median_value
                numeric_frames[column] = coerced.fillna(median_value)
            else:
                self.categorical_cols.append(column)
                series = X[column].fillna('').astype(str)
                default_value = series.mode().iloc[0] if not series.mode().empty else ''
                self.categorical_defaults[column] = default_value

        numeric_frame = pd.DataFrame(numeric_frames, index=X.index) if numeric_frames else pd.DataFrame(index=X.index)
        categorical_frame = X[self.categorical_cols].fillna(self.categorical_defaults).astype(str) if self.categorical_cols else pd.DataFrame(index=X.index)

        encoded_categorical = pd.get_dummies(categorical_frame, columns=self.categorical_cols, drop_first=False) if self.categorical_cols else pd.DataFrame(index=X.index)
        combined = pd.concat([numeric_frame, encoded_categorical], axis=1)

        self.feature_columns = combined.columns.tolist()

        if self.numeric_cols:
            self.scaler.fit(numeric_frame[self.numeric_cols])

        self.fitted = True
        return self

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError('Preprocessor must be fitted before transform()')

        X = self._prepare_features(df)

        for column in self.numeric_cols:
            if column not in X.columns:
                X[column] = self.numeric_defaults[column]
            X[column] = pd.to_numeric(X[column], errors='coerce').fillna(self.numeric_defaults[column])

        for column in self.categorical_cols:
            if column not in X.columns:
                X[column] = self.categorical_defaults[column]
            X[column] = X[column].fillna(self.categorical_defaults[column]).astype(str)

        numeric_frame = pd.DataFrame(index=X.index)
        if self.numeric_cols:
            numeric_frame = X[self.numeric_cols].copy()

        categorical_frame = pd.DataFrame(index=X.index)
        if self.categorical_cols:
            categorical_frame = pd.get_dummies(X[self.categorical_cols], columns=self.categorical_cols, drop_first=False)

        combined = pd.concat([numeric_frame, categorical_frame], axis=1)

        for column in self.feature_columns:
            if column not in combined.columns:
                combined[column] = 0

        combined = combined.reindex(columns=self.feature_columns, fill_value=0)

        if self.numeric_cols:
            combined[self.numeric_cols] = self.scaler.transform(combined[self.numeric_cols])

        return combined

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def preprocess_pipeline(df, target_col='Churn'):
    """Fit preprocessing, return processed features, target, and reusable preprocessor."""
    if target_col not in df.columns:
        raise KeyError('Expected target column `Churn` in dataframe')

    y = df[target_col].map({'No': 0, 'Yes': 1})
    if y.isna().any():
        raise ValueError('Target column `Churn` must contain only `Yes` and `No` values')

    processor = TabularPreprocessor(target_col=target_col)
    X = processor.fit_transform(df)

    return X, y.astype(int), processor
