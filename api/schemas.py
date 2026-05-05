from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CustomerData(BaseModel):
    model_config = ConfigDict(
        extra='ignore',
        json_schema_extra={
            'example': {
                'SeniorCitizen': 0,
                'tenure': 24,
                'MonthlyCharges': 65.5,
                'TotalCharges': 1570.8,
                'Contract': 'One year',
                'PhoneService': 'Yes',
                'InternetService': 'Fiber optic'
            }
        }
    )

    customerID: Optional[str] = None
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = Field(default=None, ge=0, le=1)
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[int] = Field(default=None, ge=0)
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = Field(default=None, ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float = Field(..., description='Probability of churn (0-1)')
    prediction: str = Field(..., description='Predicted class: Churn or No Churn')
    confidence: float = Field(..., description='Confidence level of prediction')


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    version: str