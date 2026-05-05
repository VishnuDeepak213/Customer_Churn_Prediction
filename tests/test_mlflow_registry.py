from pathlib import Path

import pytest

from src.mlflow_registry import register_model


def test_register_model_raises_for_missing_model_file(tmp_path: Path):
    missing = tmp_path / 'missing_model.pkl'

    with pytest.raises(FileNotFoundError):
        register_model(
            model_path=missing,
            tracking_uri='http://localhost:5000',
            experiment_name='churn-prediction',
            registered_model_name='churn-predictor',
        )
