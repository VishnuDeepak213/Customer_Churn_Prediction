import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    @staticmethod
    def create_tenure_groups(df):
        """Create tenure-based feature groups"""
        df_feat = df.copy()
        
        if 'tenure' in df_feat.columns:
            df_feat['tenure_group'] = pd.cut(
                df_feat['tenure'],
                bins=[0, 12, 24, 48, 72],
                labels=['0-1yr', '1-2yr', '2-4yr', '4yr+']
            )
            # Convert to numeric
            df_feat['tenure_group_code'] = pd.factorize(df_feat['tenure_group'])[0]
            logger.info("✅ Created tenure_group feature")
        
        return df_feat
    
    @staticmethod
    def create_monthly_charge_groups(df):
        """Create monthly charge-based groups"""
        df_feat = df.copy()
        
        if 'MonthlyCharges' in df_feat.columns:
            df_feat['monthly_charge_group'] = pd.qcut(
                df_feat['MonthlyCharges'],
                q=4,
                labels=['Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )
            df_feat['monthly_charge_group_code'] = pd.factorize(df_feat['monthly_charge_group'])[0]
            logger.info("✅ Created monthly_charge_group feature")
        
        return df_feat
    
    @staticmethod
    def create_customer_lifetime_value(df):
        """Create CLV-based feature"""
        df_feat = df.copy()
        
        if 'tenure' in df_feat.columns and 'MonthlyCharges' in df_feat.columns:
            df_feat['total_value'] = df_feat['tenure'] * df_feat['MonthlyCharges']
            df_feat['avg_value_per_month'] = df_feat['MonthlyCharges']
            logger.info("✅ Created CLV features")
        
        return df_feat
    
    @staticmethod
    def create_service_adoption_score(df):
        """Create adoption score based on services used"""
        df_feat = df.copy()
        
        # Identify service columns (yes/no services)
        service_cols = [col for col in df_feat.columns if 'PhoneService' in col or 'InternetService' in col]
        
        if service_cols:
            # Count services per customer
            df_feat['num_services'] = (df_feat[service_cols] == 'Yes').sum(axis=1)
            logger.info(f"✅ Created num_services feature")
        
        return df_feat
    
    @staticmethod
    def create_contract_risk_score(df):
        """Create risk score based on contract type"""
        df_feat = df.copy()
        
        if 'Contract' in df_feat.columns:
            risk_map = {
                'Month-to-month': 0.8,
                'One year': 0.5,
                'Two year': 0.2
            }
            df_feat['contract_risk_score'] = df_feat['Contract'].map(risk_map)
            logger.info("✅ Created contract_risk_score feature")
        
        return df_feat
    
    @staticmethod
    def engineer_all_features(df):
        """Apply all feature engineering"""
        df_feat = df.copy()
        
        df_feat = FeatureEngineer.create_tenure_groups(df_feat)
        df_feat = FeatureEngineer.create_monthly_charge_groups(df_feat)
        df_feat = FeatureEngineer.create_customer_lifetime_value(df_feat)
        df_feat = FeatureEngineer.create_service_adoption_score(df_feat)
        df_feat = FeatureEngineer.create_contract_risk_score(df_feat)
        
        logger.info(f"\\n✅ Feature engineering complete. Shape: {df_feat.shape}")
        return df_feat

# Usage
if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    
    ingestion = DataIngestion()
    df = pd.read_sql("SELECT * FROM raw_data.telco_churn", ingestion.engine)
    
    # Apply feature engineering
    df_engineered = FeatureEngineer.engineer_all_features(df)
    
    print(df_engineered.head())
    print(df_engineered.describe())