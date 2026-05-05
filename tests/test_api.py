import pytest

import api.main as api_main
from api.schemas import CustomerData


@pytest.mark.asyncio
async def test_health_endpoint():
    api_main.model_loader = object()
    response = await api_main.health_check()
    assert response.status == 'healthy'
    assert response.model_loaded is True


@pytest.mark.asyncio
async def test_predict_endpoint_accepts_partial_payload():
    class FakeLoader:
        def predict(self, _):
            return {
                'prediction': 'No Churn',
                'churn_probability': 0.23,
                'no_churn_probability': 0.77,
            }

    api_main.model_loader = FakeLoader()

    customer = CustomerData(
        SeniorCitizen=0,
        tenure=24,
        MonthlyCharges=65.5,
        TotalCharges=1570.0,
        Contract='One year',
        PhoneService='Yes',
        InternetService='Fiber optic',
        OnlineSecurity='No',
        TechSupport='No',
    )

    response = await api_main.predict(customer)
    assert 0.0 <= response.churn_probability <= 1.0
    assert response.prediction in ['Churn', 'No Churn']
