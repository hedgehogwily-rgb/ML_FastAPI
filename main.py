from fastapi import FastAPI, HTTPException, Query, Body, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dataset_service import ChurnDatasetService, EmptyDatasetError, ChurnServiceError
from model_service import (
    evaluate_churn_model,
    get_model_status,
    load_saved_model,
    persist_trained_model,
    train_churn_model,
    predict_churn,
    UnknownModelTypeError,
)
from schemas import (
    DatasetRowChurn,
    FeatureVectorChurn,
    ModelStatusResponse,
    SplitInfoResponse,
    TrainMetricsResponse,
    PredictionResponseChurn,
    TrainingConfigChurn,
    ErrorResponse,
)


app = FastAPI()
dataset_service = ChurnDatasetService("data/churn_dataset.csv")
load_saved_model()


def error_detail(code: str, message: str, details: dict | None = None) -> dict:
    return ErrorResponse(code=code, message=message, details=details or {}).model_dump()

@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}

@app.post(
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
            }
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
            ]
        }
    }
)):
    model_status = get_model_status()
    if not model_status.is_trained:
        raise HTTPException(
            status_code=400,
            detail=error_detail("model_not_trained", "Model is not trained"),
        )

    if isinstance(payload, list):
        predictions = []
        for feature_vector in payload:
            prediction = predict_churn(feature_vector)
            predictions.append(prediction)
        return predictions
    else:
        prediction = predict_churn(payload)
        return prediction


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def get_dataset_preview(limit: int = Query(5, ge=0)):
    return dataset_service.preview(limit)


@app.get("/dataset/info")
def get_dataset_info():
    return dataset_service.info()


@app.get("/dataset/split-info", response_model=SplitInfoResponse)
def get_split_info(test_size: float = Query(0.2, gt=0, lt=1), random_state: int = Query(42)):
    return dataset_service.split_info(test_size=test_size, random_state=random_state)


@app.post(
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
def train_model(config: TrainingConfigChurn = Body(
    openapi_examples={
        "logreg": {
            "summary": "Логистическая регрессия",
            "value": {
                "model_type": "logreg",
                "hyperparameters": {"C": 1.0, "max_iter": 1000, "random_state": 42},
            }
        },
    }
), test_size: float = Query(0.2, gt=0, lt=1), random_state: int = Query(42)):
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
    return metrics


@app.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    return get_model_status()


@app.get("/model/schema", response_model=dict)
def get_model_schema():
    return dataset_service.schema()



@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        content = exc.detail
    else:
        content = error_detail("http_error", str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [
        {
            "field": ".".join(str(part) for part in err["loc"]),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    specific = [item for item in errors if item["type"] != "list_type"]
    if specific:
        errors = specific
    feature_errors = {"missing", "extra_forbidden"}
    code = "invalid_feature_count" if any(item["type"] in feature_errors for item in errors) else "invalid_type"
    return JSONResponse(
        status_code=422,
        content=error_detail(code, "Request validation failed", {"errors": errors}),
    )


@app.exception_handler(ChurnServiceError)
async def churn_service_error_handler(request: Request, exc: ChurnServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_detail(exc.code, exc.message, exc.details),
    )