from contextlib import asynccontextmanager
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models_loader import get_model_loader
from api.schemas import CustomerData, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_version = '1.0.0'
model_loader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loader
    model_loader = get_model_loader()
    logger.info('Model loaded on startup')
    yield
    logger.info('Shutting down API')


app = FastAPI(
    title='Customer Churn Prediction API',
    description='Predict customer churn using the trained XGBoost model',
    version=app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/')
async def root():
    return {
        'service': 'Customer Churn Prediction API',
        'version': app_version,
        'status': 'running',
        'docs_url': '/docs',
    }


@app.get('/health', response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status='healthy',
        model_loaded=model_loader is not None,
        version=app_version,
    )


@app.get('/model-info')
async def model_info():
    return {
        'model_type': 'XGBoost Classifier',
        'target': 'Customer Churn',
        'classes': ['No Churn', 'Churn'],
        'expected_auc_roc': 0.89,
        'version': app_version,
        'deployed_at': datetime.now().isoformat(),
    }


@app.post('/predict', response_model=PredictionResponse)
async def predict(customer: CustomerData):
    try:
        if model_loader is None:
            raise HTTPException(status_code=500, detail='Model not loaded')

        result = model_loader.predict(customer.model_dump(exclude_none=True))
        churn_probability = result['churn_probability']

        return PredictionResponse(
            churn_probability=churn_probability,
            prediction=result['prediction'],
            confidence=max(churn_probability, 1 - churn_probability),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Prediction error')
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception('Unhandled error')
    return JSONResponse(status_code=500, content={'detail': 'Internal server error', 'error': str(exc)})


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')