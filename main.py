from fastapi import FastAPI
from dataset_service import ChurnDatasetService
from schemas import FeatureVectorChurn, DatasetRowChurn

app = FastAPI()
dataset_service = ChurnDatasetService("data/churn_dataset.csv")

@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}

@app.post("/predict", response_model=FeatureVectorChurn)
def predict(feature_vector: FeatureVectorChurn):
    return feature_vector


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def get_dataset_preview(limit: int = 5):
    return dataset_service.preview(limit)


@app.get("/dataset/info")
def get_dataset_info():
    return dataset_service.info()