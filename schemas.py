from pydantic import BaseModel
from typing import Any


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int
    churn: int


class SplitInfoResponse(BaseModel):
    train_size: int
    test_size: int
    train_churn_distribution: dict[str, float]
    test_churn_distribution: dict[str, float]


class TrainMetricsResponse(BaseModel):
    accuracy: float
    f1: float


class ModelStatusResponse(BaseModel):
    is_trained: bool
    trained_at: str | None
    accuracy: float | None
    f1: float | None
    model_type: str | None
    hyperparameters: dict[str, Any] | None


class PredictionResponseChurn(BaseModel):
    prediction: int
    probabilities: dict[str, float]


class TrainingConfigChurn(BaseModel):
    model_type: str
    hyperparameters: dict[str, Any]