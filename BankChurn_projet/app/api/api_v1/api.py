from fastapi import APIRouter

from .endpoints import churn_model_one_prediction
from .endpoints import churn_model_multi_predictions

api_router = APIRouter()
api_router.include_router(churn_model_one_prediction.router_model_one_prediction, prefix = "/churn_model_one_prediction", tags = ["churn_model_one_prediction"])
api_router.include_router(churn_model_multi_predictions.router_model_multi_predictions, prefix = "/churn_model_multi_predictions", tags = ["churn_model_multi_predictions"])

