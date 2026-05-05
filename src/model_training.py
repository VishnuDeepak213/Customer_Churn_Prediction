import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


class ModelTrainer:
    def __init__(self):
        self.model = None

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost model and return model + metrics dict.
        """
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        self.model.fit(X_train, y_train, verbose=False)

        # Predict
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)

        # Compute metrics
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }

        return self.model, metrics
