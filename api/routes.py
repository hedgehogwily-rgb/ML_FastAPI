import logging

from fastapi import APIRouter, Body, HTTPException, Query

from core.config import DATASET_PATH
from core.errors import error_detail
from ml.dataset import ChurnDatasetService, EmptyDatasetError
from ml.pipeline import (
    UnknownModelTypeError,
    evaluate_churn_model,
    get_model_metrics,
    get_model_status,
    persist_trained_model,
    predict_churn,
    train_churn_model,
)
from schemas import (
    DatasetRowChurn,
    ErrorResponse,
    FeatureVectorChurn,
    ModelStatusResponse,
    PredictionResponseChurn,
    SplitInfoResponse,
    TrainingConfigChurn,
    TrainMetricsResponse,
)

logger = logging.getLogger("churn")
router = APIRouter()
dataset_service = ChurnDatasetService(DATASET_PATH)


@router.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@router.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    responses={
        200: {
            "description": "Предсказанный класс и вероятности классов",
            "content": {
                "application/json": {
                    "examples": {
                        "one_client": {
                            "summary": "Ответ для одного клиента",
                            "value": {
                                "prediction": 0,
                                "probabilities": {"0": 0.83, "1": 0.17},
                            },
                        },
                        "several_clients": {
                            "summary": "Ответ для списка",
                            "value": [
                                {"prediction": 1, "probabilities": {"0": 0.31, "1": 0.69}},
                                {"prediction": 0, "probabilities": {"0": 0.91, "1": 0.09}},
                            ],
                        },
                    }
                }
            },
        },
        400: {
            "description": "Модель ещё не обучена или тело запроса не прошло проверку",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "model_not_trained": {
                            "summary": "Модель не обучена",
                            "value": {
                                "code": "model_not_trained",
                                "message": "Model is not trained",
                                "details": {},
                            },
                        },
                        "invalid_type": {
                            "summary": "Неверный тип поля",
                            "value": {
                                "code": "invalid_type",
                                "message": "Request validation failed",
                                "details": {
                                    "errors": [
                                        {
                                            "field": "body.monthly_fee",
                                            "message": "Input should be a valid number",
                                            "type": "float_type",
                                        }
                                    ]
                                },
                            },
                        },
                        "invalid_feature_count": {
                            "summary": "Неверное число признаков",
                            "value": {
                                "code": "invalid_feature_count",
                                "message": "Request validation failed",
                                "details": {
                                    "errors": [
                                        {
                                            "field": "body.region",
                                            "message": "Field required",
                                            "type": "missing",
                                        }
                                    ]
                                },
                            },
                        },
                    }
                }
            },
        },
    },
)
def predict(payload: list[FeatureVectorChurn] | FeatureVectorChurn = Body(
    openapi_examples={
        "one_client": {
            "summary": "Один клиент",
            "value": {
                "monthly_fee": 9.99,
                "usage_hours": 27.92,
                "support_requests": 1,
                "account_age_months": 14,
                "failed_payments": 1,
                "region": "america",
                "device_type": "desktop",
                "payment_method": "card",
                "autopay_enabled": 1,
            },
        },
        "multiple_clients": {
            "summary": "Несколько клиентов",
            "value": [
                {
                    "monthly_fee": 9.99,
                    "usage_hours": 4.0,
                    "support_requests": 6,
                    "account_age_months": 2,
                    "failed_payments": 4,
                    "region": "asia",
                    "device_type": "mobile",
                    "payment_method": "crypto",
                    "autopay_enabled": 0,
                },
                {
                    "monthly_fee": 29.99,
                    "usage_hours": 40.0,
                    "support_requests": 0,
                    "account_age_months": 24,
                    "failed_payments": 0,
                    "region": "europe",
                    "device_type": "desktop",
                    "payment_method": "card",
                    "autopay_enabled": 1,
                },
            ],
        },
    }
)):
    client_count = len(payload) if isinstance(payload, list) else 1
    logger.info("predict called clients=%s", client_count)
    if not get_model_status().is_trained:
        raise HTTPException(
            status_code=400,
            detail=error_detail("model_not_trained", "Model is not trained"),
        )

    if isinstance(payload, list):
        return [predict_churn(feature_vector) for feature_vector in payload]
    return predict_churn(payload)


@router.get("/dataset/preview", response_model=list[DatasetRowChurn])
def get_dataset_preview(limit: int = Query(5, ge=0)):
    return dataset_service.preview(limit)


@router.get("/dataset/info")
def get_dataset_info():
    return dataset_service.info()


@router.get("/dataset/split-info", response_model=SplitInfoResponse)
def get_split_info(test_size: float = Query(0.2, gt=0, lt=1), random_state: int = Query(42)):
    return dataset_service.split_info(test_size=test_size, random_state=random_state)


@router.post(
    "/model/train",
    response_model=TrainMetricsResponse,
    responses={
        400: {
            "description": "Данные или конфигурация не подходят для обучения",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "empty_dataset": {
                            "summary": "Пустой датасет",
                            "value": {
                                "code": "empty_dataset",
                                "message": "Dataset is empty",
                                "details": {},
                            },
                        },
                        "unknown_model_type": {
                            "summary": "Неизвестный тип модели",
                            "value": {
                                "code": "unknown_model_type",
                                "message": "Unknown model type: svm",
                                "details": {},
                            },
                        },
                    }
                }
            },
        },
        404: {
            "description": "Файл датасета не найден",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "code": "file_not_found",
                        "message": "Dataset file is not loaded",
                        "details": {},
                    }
                }
            },
        },
    },
)
def train_model(
    config: TrainingConfigChurn = Body(
        openapi_examples={
            "logreg": {
                "summary": "Логистическая регрессия",
                "value": {
                    "model_type": "logreg",
                    "hyperparameters": {"C": 1.0, "max_iter": 1000, "random_state": 42},
                },
            },
        }
    ),
    test_size: float = Query(0.2, gt=0, lt=1),
    random_state: int = Query(42),
):
    try:
        train_data, test_data = dataset_service.split_data(
            test_size=test_size,
            random_state=random_state,
        )
        pipeline = train_churn_model(config=config, train_data=train_data)
        metrics = evaluate_churn_model(pipeline, test_data)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=error_detail("file_not_found", "Dataset file is not loaded"),
        )
    except EmptyDatasetError as exc:
        raise HTTPException(
            status_code=400,
            detail=error_detail("empty_dataset", str(exc)),
        )
    except UnknownModelTypeError as exc:
        raise HTTPException(
            status_code=400,
            detail=error_detail("unknown_model_type", str(exc)),
        )
    except TypeError as exc:
        raise HTTPException(
            status_code=400,
            detail=error_detail("invalid_hyperparameters", str(exc)),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=error_detail("invalid_training_data", str(exc)),
        )

    persist_trained_model(pipeline, metrics, config)
    logger.info(
        "model trained type=%s accuracy=%.4f f1=%.4f roc_auc=%.4f",
        config.model_type,
        metrics.accuracy,
        metrics.f1,
        metrics.roc_auc,
    )
    return metrics


@router.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    return get_model_status()


@router.get("/model/schema")
def get_model_schema():
    return dataset_service.schema()


@router.get("/model/metrics")
def model_metrics(
    model_type: str | None = Query(None),
    limit: int = Query(10, ge=1),
):
    return get_model_metrics(model_type=model_type, limit=limit)


@router.get("/health")
def health():
    return {
        "model_available": get_model_status().is_trained,
        "dataset_loaded": dataset_service.csv_path.is_file(),
    }
