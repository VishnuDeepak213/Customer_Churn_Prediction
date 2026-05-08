import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os

# Base directory: works both locally and on Streamlit Cloud
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_resource
def load_model_and_data():
    """Load model and dataset — paths work on Streamlit Cloud"""

    # On Streamlit Cloud, repo root is /mount/src/<repo_name>
    # os.path resolution handles both local and cloud correctly
    model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.pkl')

    # Try multiple CSV name variants (handle casing differences)
    csv_candidates = [
        os.path.join(base_dir, 'data', 'raw', 'Telco_Churn.csv'),
        os.path.join(base_dir, 'data', 'raw', 'telco_churn.csv'),
        os.path.join(base_dir, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
    ]

    # Validate model files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            "Make sure 'models/best_model.pkl' is committed to your GitHub repo."
        )
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor not found at: {preprocessor_path}\n"
            "Make sure 'models/preprocessor.pkl' is committed to your GitHub repo."
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Load CSV — try each candidate path
    df = None
    for csv_path in csv_candidates:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            break

    if df is None:
        raise FileNotFoundError(
            "Dataset CSV not found. Make sure one of these exists in your repo:\n"
            + "\n".join(f"  - {p}" for p in csv_candidates)
        )

    # Standardise Churn column to Yes/No strings
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({1: 'Yes', 0: 'No', 'Yes': 'Yes', 'No': 'No'}).fillna(df['Churn'])

    return model, preprocessor, df


def make_prediction(customer_data, model, preprocessor):
    """Make prediction for a single customer dict"""
    input_df = pd.DataFrame([customer_data])
    X_processed = preprocessor.transform(input_df)

    pred = model.predict(X_processed)[0]
    prob = model.predict_proba(X_processed)[0]

    return {
        'prediction': 'CHURN' if pred == 1 else 'NO CHURN',
        'churn_prob': float(prob[1]),
        'no_churn_prob': float(prob[0])
    }


def get_model_metrics(model, X_test, y_test):
    """Calculate standard classification metrics"""
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        precision_score, recall_score
    )

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'F1-Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }


def detect_data_drift(df_reference, df_current, threshold=0.05):
    """Detect data drift using Kolmogorov-Smirnov test (scipy only — no evidently needed)"""
    from scipy.stats import ks_2samp

    drift_report = {}
    numerical_cols = df_reference.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        try:
            if col not in df_current.columns:
                continue

            ref_data = df_reference[col].dropna()
            cur_data = df_current[col].dropna()

            if len(ref_data) < 2 or len(cur_data) < 2:
                continue

            ref_data = ref_data[np.isfinite(ref_data)]
            cur_data = cur_data[np.isfinite(cur_data)]

            if len(ref_data) < 2 or len(cur_data) < 2:
                continue

            statistic, p_value = ks_2samp(ref_data, cur_data)

            if np.isnan(p_value):
                continue

            drift_report[col] = {
                'p_value': float(p_value),
                'is_drift': bool(p_value < threshold),
                'statistic': float(statistic)
            }
        except Exception:
            continue

    return drift_report


def get_churn_risk_segments(df, model, preprocessor):
    """Segment customers by predicted churn risk"""
    X_processed = preprocessor.transform(df)
    churn_probs = model.predict_proba(X_processed)[:, 1]

    df = df.copy()
    df['churn_probability'] = churn_probs
    df['risk_segment'] = pd.cut(
        churn_probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    return df
