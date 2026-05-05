import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    def confusion_matrix_analysis(self, save_path=None):
        """Generate and visualize confusion matrix"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Detailed metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        logger.info(f"\\nConfusion Matrix Analysis:")
        logger.info(f"True Negatives: {tn}, False Positives: {fp}")
        logger.info(f"False Negatives: {fn}, True Positives: {tp}")
        logger.info(f"Sensitivity (Recall): {sensitivity:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        
        return cm
    
    def classification_report_detailed(self):
        """Generate detailed classification report"""
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=['No Churn', 'Churn'],
            output_dict=False
        )
        
        logger.info(f"\\nClassification Report:\\n{report}")
        return report
    
    def roc_auc_analysis(self, save_path=None):
        """Generate ROC-AUC curve"""
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Visualization
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        logger.info(f"\\nROC-AUC Score: {roc_auc:.4f}")
        
        return roc_auc, fpr, tpr, thresholds
    
    def precision_recall_analysis(self, save_path=None):
        """Generate Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Visualization
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        logger.info(f"\\nPrecision-Recall AUC: {pr_auc:.4f}")
        
        return pr_auc, precision, recall
    
    def calibration_analysis(self, save_path=None):
        """Analyze and calibrate model probabilities"""
        # Plot calibration curve
        prob_true, prob_pred = calibration_curve(
            self.y_test, self.y_pred_proba, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        logger.info(f"\\n✅ Calibration analysis complete")
        
        return prob_true, prob_pred
    
    def calibrate_model(self):
        """Calibrate model using Platt scaling"""
        # Use part of test set for calibration
        calibrated_model = CalibratedClassifierCV(
            self.model, method='sigmoid', cv='prefit'
        )
        
        logger.info(f"✅ Model calibrated using Platt scaling")
        return calibrated_model
    
    def threshold_analysis(self):
        """Analyze different probability thresholds"""
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        
        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"\\nThreshold Analysis:")
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"At optimal threshold: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")
        
        return optimal_threshold, tpr, fpr

# Usage
if __name__ == "__main__":
    import joblib
    
    model = joblib.load('models/best_model.pkl')
    from sklearn.model_selection import train_test_split
    from src.data_ingestion import DataIngestion
    from src.preprocessing import preprocess_pipeline
    
    # Load test data
    ingestion = DataIngestion()
    df = pd.read_sql("SELECT * FROM raw_data.telco_churn", ingestion.engine)
    X, y, _ = preprocess_pipeline(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Evaluate
    evaluator = ModelEvaluator(model, X_test, y_test)
    evaluator.confusion_matrix_analysis()
    evaluator.classification_report_detailed()
    evaluator.roc_auc_analysis()
    evaluator.precision_recall_analysis()
    evaluator.calibration_analysis()