import logging
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, model_path=None, preprocessor_path=None):
        base_dir = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else base_dir / 'models' / 'best_model.pkl'
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else base_dir / 'models' / 'preprocessor.pkl'
        self.model = None
        self.preprocessor = None
        self.load_model()

    def load_model(self):
        """Load trained model and preprocessor."""
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info('Loaded model from %s', self.model_path)
            logger.info('Loaded preprocessor from %s', self.preprocessor_path)
        except FileNotFoundError as exc:
            logger.error('Model file not found: %s', exc)
            raise
        except Exception as exc:
            logger.error('Error loading model: %s', exc)
            raise

    def preprocess_input(self, data_dict):
        if isinstance(data_dict, dict):
            frame = pd.DataFrame([data_dict])
        else:
            frame = data_dict

        for name, transformer, columns in getattr(self.preprocessor, 'transformers_', []):
            if name == 'remainder' or not columns:
                continue

            for column in columns:
                if column not in frame.columns:
                    frame[column] = 0 if name == 'num' else 'Unknown'

                if name == 'num':
                    frame[column] = pd.to_numeric(frame[column], errors='coerce').fillna(0)
                else:
                    frame[column] = frame[column].fillna('Unknown').astype(str)

        expected_columns = getattr(self.preprocessor, 'feature_names_in_', None)
        if expected_columns is not None:
            frame = frame.reindex(columns=list(expected_columns))

        return self.preprocessor.transform(frame)

    def predict(self, data_dict):
        processed = self.preprocess_input(data_dict)
        prediction = self.model.predict(processed)[0]
        probability = self.model.predict_proba(processed)[0]

        return {
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'churn_probability': float(probability[1]),
            'no_churn_probability': float(probability[0])
        }


_model_loader = None


def get_model_loader():
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader