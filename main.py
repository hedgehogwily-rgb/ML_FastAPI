from fastapi import FastAPI, HTTPException, Query
from dataset_service import ChurnDatasetService, EmptyDatasetError
from model_service import (
    evaluate_churn_model,
    get_model_status,
    load_saved_model,
    persist_trained_model,
    train_churn_model,
)
from schemas import (
    DatasetRowChurn,
    FeatureVectorChurn,
    ModelStatusResponse,
    SplitInfoResponse,
    TrainMetricsResponse,
)

app = FastAPI()
dataset_service = ChurnDatasetService("data/churn_dataset.csv")
load_saved_model()

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


@app.post("/model/train", response_model=TrainMetricsResponse)
def train_model(test_size: float = Query(0.2, gt=0, lt=1), random_state: int = Query(42)):
    try:
        train_data, test_data = dataset_service.split_data(
            test_size=test_size,
            random_state=random_state,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file is not loaded")
    except EmptyDatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    pipeline = train_churn_model(train_data)
    metrics = evaluate_churn_model(pipeline, test_data)
    persist_trained_model(pipeline, metrics)
    return metrics


@app.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    return get_model_status()