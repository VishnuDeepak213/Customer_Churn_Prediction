import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X_train)
        logger.info("SHAP explainer initialized")

    def _class_1_shap_values(self):
        if isinstance(self.shap_values, list):
            return self.shap_values[1]
        return self.shap_values

    def summary_plot(self, plot_type="bar", max_display=15, save_path=None):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            self.X_train,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=save_path is None,
        )
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        logger.info("SHAP summary plot generated")

    def waterfall_plot(self, instance_idx=0, save_path=None):
        shap_vals = self._class_1_shap_values()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[instance_idx],
                base_values=self.explainer.expected_value,
                data=self.X_train.iloc[instance_idx],
                feature_names=self.feature_names,
            )
        )
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        logger.info("SHAP waterfall plot generated for instance %s", instance_idx)

    def beeswarm_plot(self, max_display=15, save_path=None):
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_train,
            feature_names=self.feature_names,
            plot_type="violin",
            max_display=max_display,
            show=save_path is None,
        )
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        logger.info("SHAP beeswarm plot generated")

    def force_plot(self, instance_idx=0):
        shap_vals = self._class_1_shap_values()
        shap.force_plot(
            self.explainer.expected_value,
            shap_vals[instance_idx],
            self.X_train.iloc[instance_idx],
            feature_names=self.feature_names,
        )
        logger.info("SHAP force plot generated for instance %s", instance_idx)

    def top_features(self, n_features=10):
        shap_vals = self._class_1_shap_values()
        feature_importance = np.mean(np.abs(shap_vals), axis=0)
        top_idx = np.argsort(feature_importance)[-n_features:][::-1]
        top_features_df = pd.DataFrame(
            {
                "Feature": [self.feature_names[i] for i in top_idx],
                "Importance": feature_importance[top_idx],
            }
        )
        logger.info("Top %s features:\n%s", n_features, top_features_df)
        return top_features_df


if __name__ == "__main__":
    model = joblib.load("models/best_model.pkl")
    from src.data_ingestion import DataIngestion
    from src.preprocessing import preprocess_pipeline

    ingestion = DataIngestion()
    df = pd.read_sql("SELECT * FROM raw_data.telco_churn", ingestion.engine)
    X, y, _ = preprocess_pipeline(df)
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    shap_explainer = SHAPExplainer(model, X_train[:100])
    shap_explainer.summary_plot(plot_type="bar")
    shap_explainer.beeswarm_plot()
    shap_explainer.waterfall_plot(instance_idx=0)
    print(shap_explainer.top_features(n_features=10))
