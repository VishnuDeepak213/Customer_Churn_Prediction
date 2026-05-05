from pathlib import Path

from api.models_loader import ModelLoader


def test_model_artifacts_exist():
    assert Path('models/best_model.pkl').exists()
    assert Path('models/preprocessor.pkl').exists()


def test_model_loader_can_predict_minimal_input():
    loader = ModelLoader()
    result = loader.predict(
        {
            'SeniorCitizen': 0,
            'tenure': 24,
            'MonthlyCharges': 65.5,
            'TotalCharges': 1570.0,
            'Contract': 'One year',
            'PhoneService': 'Yes',
            'InternetService': 'Fiber optic',
        }
    )

    assert 'churn_probability' in result
    assert 0.0 <= result['churn_probability'] <= 1.0
