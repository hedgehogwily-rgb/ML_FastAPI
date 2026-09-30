import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset_service import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PreparedData
from schemas import FeatureVectorChurn, ModelStatusResponse, PredictionResponseChurn, TrainMetricsResponse

MODEL_PATH = "models/churn_model.pkl"
INFO_PATH = "models/churn_model_info.pkl"

_pipeline: Pipeline | None = None
_trained_at: str | None = None
_accuracy: float | None = None
_f1: float | None = None


def build_churn_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    if numeric_features is None:
        numeric_features = list(NUMERIC_FEATURES)
    if categorical_features is None:
        categorical_features = list(CATEGORICAL_FEATURES)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
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
    global _pipeline, _trained_at, _accuracy, _f1
    _pipeline = None
    _trained_at = None
    _accuracy = None
    _f1 = None


def _store_loaded_model(
    pipeline: Pipeline,
    trained_at: str | None,
    accuracy: float | None,
    f1: float | None,
) -> None:
    global _pipeline, _trained_at, _accuracy, _f1
    _pipeline = pipeline
    _trained_at = trained_at
    _accuracy = accuracy
    _f1 = f1


def save_model_info(accuracy: float, f1: float, trained_at: str, path: str = INFO_PATH) -> None:
    _ensure_models_dir()
    with open(path, "wb") as f:
        pickle.dump(
            {"trained_at": trained_at, "accuracy": accuracy, "f1": f1},
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


def persist_trained_model(pipeline: Pipeline, metrics: TrainMetricsResponse) -> None:
    trained_at = datetime.now().isoformat(timespec="seconds")
    save_churn_model(pipeline)
    save_model_info(metrics.accuracy, metrics.f1, trained_at)
    _store_loaded_model(pipeline, trained_at, metrics.accuracy, metrics.f1)


def load_saved_model() -> None:
    try:
        pipeline = load_churn_model()
    except FileNotFoundError:
        _clear_loaded_model()
        return

    info = load_model_info()
    if info is None:
        _store_loaded_model(pipeline, None, None, None)
        return

    _store_loaded_model(
        pipeline,
        str(info["trained_at"]),
        float(info["accuracy"]),
        float(info["f1"]),
    )


def get_model_status() -> ModelStatusResponse:
    return ModelStatusResponse(
        is_trained=_pipeline is not None,
        trained_at=_trained_at,
        accuracy=_accuracy,
        f1=_f1,
    )


def train_churn_model(train_data: PreparedData) -> Pipeline:
    pipeline = build_churn_pipeline(
        numeric_features=train_data.numeric_features,
        categorical_features=train_data.categorical_features,
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
    row = pd.DataFrame([feature_vector.model_dump()])
    predicted = int(pipeline.predict(row)[0])
    proba = pipeline.predict_proba(row)[0]
    classes = pipeline.named_steps["classifier"].classes_
    probabilities = {str(label): float(value) for label, value in zip(classes, proba)}
    return PredictionResponseChurn(prediction=predicted, probabilities=probabilities)
