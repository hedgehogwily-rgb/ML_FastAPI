from fastapi import FastAPI, Query
from dataset_service import ChurnDatasetService
from schemas import FeatureVectorChurn, DatasetRowChurn, SplitInfoResponse

app = FastAPI()
dataset_service = ChurnDatasetService("data/churn_dataset.csv")

@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}

@app.post("/predict", response_model=FeatureVectorChurn)
def predict(feature_vector: FeatureVectorChurn):
    return feature_vector


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def get_dataset_preview(limit: int = Query(5, ge=0)):
    return dataset_service.preview(limit)


@app.get("/dataset/info")
def get_dataset_info():
    return dataset_service.info()


@app.get("/dataset/split-info", response_model=SplitInfoResponse)
def get_split_info(test_size: float = Query(0.2, gt=0, lt=1), random_state: int = Query(42)):
    return dataset_service.split_info(test_size=test_size, random_state=random_state)