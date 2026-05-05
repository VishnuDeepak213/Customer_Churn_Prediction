from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImbalanceHandler:
    @staticmethod
    def apply_smote(X, y, sampling_strategy=0.5, random_state=42):
        """
        Apply SMOTE to handle class imbalance
        
        Args:
            X: Features
            y: Target
            sampling_strategy: Ratio of minority to majority (0.5 = 50%)
            random_state: Random seed
        
        Returns:
            X_balanced, y_balanced
        """
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        logger.info(f"\\nOriginal distribution:\\n{y.value_counts()}")
        logger.info(f"\\nAfter SMOTE:\\n")
        import pandas as pd
        balanced_dist = pd.Series(y_balanced).value_counts()
        logger.info(f"{balanced_dist}")
        
        return X_balanced, y_balanced
    
    @staticmethod
    def apply_combined_strategy(X, y, over_sampling=0.5, random_state=42):
        """Apply combined over-sampling and under-sampling"""
        pipeline = ImbPipeline([
            ('over', SMOTE(sampling_strategy=over_sampling, random_state=random_state)),
            ('under', RandomUnderSampler(sampling_strategy=0.8, random_state=random_state))
        ])
        
        X_balanced, y_balanced = pipeline.fit_resample(X, y)
        logger.info(f"✅ Applied combined resampling strategy")
        
        return X_balanced, y_balanced

# Usage in training pipeline
if __name__ == "__main__":
    import pandas as pd
    from src.data_ingestion import DataIngestion
    from src.preprocessing import preprocess_pipeline
    
    # Load and preprocess
    ingestion = DataIngestion()
    df = pd.read_sql("SELECT * FROM raw_data.telco_churn", ingestion.engine)
    X, y, processor = preprocess_pipeline(df)
    
    # Apply SMOTE
    X_balanced, y_balanced = ImbalanceHandler.apply_smote(X, y)