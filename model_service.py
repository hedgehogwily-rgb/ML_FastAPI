import math
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from typing import Any
from dataset_service import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ChurnServiceError, PreparedData
from schemas import FeatureVectorChurn, ModelStatusResponse, PredictionResponseChurn, TrainMetricsResponse, TrainingConfigChurn

MODEL_PATH = "models/churn_model.pkl"
INFO_PATH = "models/churn_model_info.pkl"

_pipeline: Pipeline | None = None
_trained_at: str | None = None
_accuracy: float | None = None
_f1: float | None = None
_model_type: str | None = None
_hyperparameters: dict[str, Any] | None = None


class UnknownModelTypeError(Exception):
    pass


def build_classifier(config: TrainingConfigChurn):
    if config.model_type == "logreg":
        return LogisticRegression(**config.hyperparameters)
    if config.model_type == "random_forest":
        return RandomForestClassifier(**config.hyperparameters)
    raise UnknownModelTypeError(f"Unknown model type: {config.model_type}")


def build_churn_pipeline(
    config: TrainingConfigChurn,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", build_classifier(config)),
        ]
    )


def _ensure_models_dir() -> None:
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)


def save_churn_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    _ensure_models_dir()
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)


def load_churn_model(path: str = MODEL_PATH) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def _clear_loaded_model() -> None:
    global _pipeline, _trained_at, _accuracy, _f1, _model_type, _hyperparameters
    _pipeline = None
    _trained_at = None
    _accuracy = None
    _f1 = None
    _model_type = None
    _hyperparameters = None

def _store_loaded_model(
    pipeline: Pipeline,
    trained_at: str | None,
    accuracy: float | None,
    f1: float | None,
    model_type: str | None,
    hyperparameters: dict[str, Any] | None,
) -> None:
    global _pipeline, _trained_at, _accuracy, _f1, _model_type, _hyperparameters
    _pipeline = pipeline
    _trained_at = trained_at
    _accuracy = accuracy
    _f1 = f1
    _model_type = model_type
    _hyperparameters = hyperparameters


def save_model_info(accuracy: float, f1: float, trained_at: str, model_type: str, hyperparameters: dict[str, Any], path: str = INFO_PATH) -> None:
    _ensure_models_dir()
    with open(path, "wb") as f:
        pickle.dump(
            {"trained_at": trained_at, "accuracy": accuracy, "f1": f1, "model_type": model_type, "hyperparameters": hyperparameters},
            f,
        )


def load_model_info(path: str = INFO_PATH) -> dict | None:
    try:
        with open(path, "rb") as f:
            info = pickle.load(f)
    except FileNotFoundError:
        return None
    if not isinstance(info, dict) or "pipeline" in info:
        return None
    return info


def persist_trained_model(pipeline: Pipeline, metrics: TrainMetricsResponse, config: TrainingConfigChurn) -> None:
    trained_at = datetime.now().isoformat(timespec="seconds")
    save_churn_model(pipeline)
    save_model_info(metrics.accuracy, metrics.f1, trained_at, config.model_type, config.hyperparameters)
    _store_loaded_model(pipeline, trained_at, metrics.accuracy, metrics.f1, config.model_type, config.hyperparameters)


def load_saved_model() -> None:
    try:
        pipeline = load_churn_model()
    except FileNotFoundError:
        _clear_loaded_model()
        return

    info = load_model_info()
    if info is None:
        _store_loaded_model(pipeline, None, None, None, None, None)
        return

    _store_loaded_model(
        pipeline,
        str(info["trained_at"]),
        float(info["accuracy"]),
        float(info["f1"]),
        info.get("model_type"),
        info.get("hyperparameters"),
    )


def get_model_status() -> ModelStatusResponse:
    return ModelStatusResponse(
        is_trained=_pipeline is not None,
        trained_at=_trained_at,
        accuracy=_accuracy,
        f1=_f1,
        model_type=_model_type,
        hyperparameters=_hyperparameters,
    )


def train_churn_model(config: TrainingConfigChurn, train_data: PreparedData) -> Pipeline:
    pipeline = build_churn_pipeline(
        config=config,
    )
    pipeline.fit(train_data.X, train_data.y)
    return pipeline


def evaluate_churn_model(pipeline: Pipeline, test_data: PreparedData) -> TrainMetricsResponse:
    y_pred = pipeline.predict(test_data.X)
    return TrainMetricsResponse(
        accuracy=float(accuracy_score(test_data.y, y_pred)),
        f1=float(f1_score(test_data.y, y_pred, zero_division=0)),
    )

def predict_churn(feature_vector: FeatureVectorChurn) -> PredictionResponseChurn:
    pipeline = _pipeline
    if pipeline is None:
        raise ChurnServiceError("model_not_trained", "Model is not trained")

    row = pd.DataFrame([feature_vector.model_dump()])
    numeric = row.select_dtypes(include="number")
    non_finite = [
        column
        for column in numeric.columns
        if not bool(numeric[column].map(math.isfinite).all())
    ]
    if non_finite:
        raise ChurnServiceError(
            "invalid_type",
            "Numeric features must be finite numbers",
            details={"fields": non_finite},
        )

    try:
        predicted = int(pipeline.predict(row)[0])
        proba = pipeline.predict_proba(row)[0]
    except ValueError as exc:
        raise ChurnServiceError("prediction_failed", str(exc)) from exc

    classes = pipeline.named_steps["classifier"].classes_
    probabilities = {str(label): float(value) for label, value in zip(classes, proba)}
    return PredictionResponseChurn(prediction=predicted, probabilities=probabilities)
