from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

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

@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}

@app.post("/predict", response_model=FeatureVectorChurn)
def predict(feature_vector: FeatureVectorChurn):
    return feature_vector