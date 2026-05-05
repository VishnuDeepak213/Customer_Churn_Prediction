import pandas as pd
import numpy as np
import streamlit as st
import joblib
import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

@st.cache_resource
def load_model_and_data():
    """Load model and dataset"""
    model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.pkl')
    csv_path = os.path.join(base_dir, 'data', 'raw', 'Telco_Churn.csv')
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Load dataset from CSV
    df = pd.read_csv(csv_path)
    
    return model, preprocessor, df

def make_prediction(customer_data, model, preprocessor):
    """Make prediction for customer data"""
    import pandas as pd
    
    df = pd.DataFrame([customer_data])
    X_processed = preprocessor.transform(df)
    
    pred = model.predict(X_processed)[0]
    prob = model.predict_proba(X_processed)[0]
    
    return {
        'prediction': 'CHURN' if pred == 1 else 'NO CHURN',
        'churn_prob': prob[1],
        'no_churn_prob': prob[0]
    }

def get_model_metrics(model, X_test, y_test):
    """Calculate model metrics"""
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
    """Detect data drift using Kolmogorov-Smirnov test"""
    from scipy.stats import ks_2samp
    
    drift_report = {}
    
    # Compare numerical features
    numerical_cols = df_reference.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        try:
            if col not in df_current.columns:
                continue
            
            # Get clean data (remove NaN and inf values)
            ref_data = df_reference[col].dropna()
            cur_data = df_current[col].dropna()
            
            # Skip if insufficient data
            if len(ref_data) < 2 or len(cur_data) < 2:
                continue
            
            # Remove infinite values
            ref_data = ref_data[np.isfinite(ref_data)]
            cur_data = cur_data[np.isfinite(cur_data)]
            
            if len(ref_data) < 2 or len(cur_data) < 2:
                continue
            
            statistic, p_value = ks_2samp(ref_data, cur_data)
            
            # Handle NaN p-value
            if np.isnan(p_value):
                continue
            
            is_drift = p_value < threshold
            drift_report[col] = {
                'p_value': float(p_value),
                'is_drift': bool(is_drift),
                'statistic': float(statistic)
            }
        except Exception as col_err:
            # Skip columns that fail
            continue
    
    return drift_report

def get_churn_risk_segments(df, model, preprocessor):
    """Segment customers by churn risk"""
    X_processed = preprocessor.transform(df)
    churn_probs = model.predict_proba(X_processed)[:, 1]
    
    df['churn_probability'] = churn_probs
    df['risk_segment'] = pd.cut(
        churn_probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    return df